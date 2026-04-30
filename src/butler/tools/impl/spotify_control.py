from __future__ import annotations

import functools
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext
from butler.voice.normalize import normalize_text, similarity, token_overlap

logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

BUTLER_SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "user-read-recently-played",
    "user-library-read",
    "user-library-modify",
    "playlist-read-private",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-top-read",
    "user-read-private",          # Required to fetch user country (market)
    "user-follow-read",
    "user-follow-modify",
    "ugc-image-upload",
    "user-read-playback-position",
])

_RE_COMMAND_PREFIX = re.compile(
    r"^\s*(?:play|put on|start|start playing|listen to)\s+", re.IGNORECASE
)

# Module-level market cache — resets on process restart, safe for single-user desktop app
_market_cache: dict[str, str] = {}

# ── RETRY DECORATOR ───────────────────────────────────────────────────────────

def spotify_retry(max_retries: int = 3, base_wait: int = 5):
    """
    Production-grade retry decorator for all Spotify API calls.
    Handles: 429 (rate limit), 401 (token), 403 (Premium/scope), 404 (no device), 5xx (server).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import spotipy
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except spotipy.exceptions.SpotifyException as e:
                    if e.http_status == 429:
                        retry_after = e.headers.get("Retry-After") if e.headers else None
                        wait = (
                            int(retry_after or base_wait) + 1
                            if attempt == 0 else (2 ** attempt) * base_wait
                        )
                        logger.warning("[429] Rate limited on %s. Waiting %ss.", func.__name__, wait)
                        time.sleep(wait)
                    elif e.http_status == 401:
                        logger.error("[401] Token expired in %s. Re-raising.", func.__name__)
                        raise
                    elif e.http_status == 403:
                        msg = str(e.msg or "")
                        if "Premium" in msg:
                            raise PermissionError("Spotify Premium is required for this action.")
                        raise PermissionError(
                            f"Spotify scope error in {func.__name__}: {msg}. "
                            "Check that the required scope was requested during authorization."
                        )
                    elif e.http_status == 404:
                        msg = str(e.msg or "")
                        if "NO_ACTIVE_DEVICE" in msg or "no active device" in msg.lower():
                            raise RuntimeError(
                                "No active Spotify device found. "
                                "Please open Spotify on your desktop or phone first."
                            )
                        raise
                    elif e.http_status in (500, 502, 503):
                        wait = (2 ** attempt) * base_wait
                        logger.warning(
                            "[%s] Spotify server error on %s. Retry in %ss.",
                            e.http_status, func.__name__, wait,
                        )
                        time.sleep(wait)
                    else:
                        raise
                    if attempt == max_retries - 1:
                        raise
            return None
        return wrapper
    return decorator


# ── SPOTIFY CLIENT + MARKET ───────────────────────────────────────────────────

def _get_spotify_client(ctx: ToolContext):
    """Build and return an authenticated Spotipy client using config credentials."""
    client_id = ctx.config.spotify_client_id
    client_secret = ctx.config.spotify_client_secret
    if not client_id or not client_secret:
        raise ValueError(
            "Spotify Client ID or Client Secret is missing from the configuration. "
            "Please add them to your .butler/config.json file."
        )
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri="http://127.0.0.1:8080",
        scope=BUTLER_SCOPES,
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _get_market(sp: Any) -> str:
    """
    Lazily fetch and cache the user's country code (e.g. 'IN' for India).
    This is THE critical fix for Indian/Punjabi artist support — passing the
    user's real market instead of a hardcoded 'US' unlocks the full India catalog.
    """
    if "market" not in _market_cache:
        try:
            user = sp.current_user()
            _market_cache["market"] = user.get("country", "IN")
        except Exception:
            _market_cache["market"] = "IN"
    return _market_cache["market"]


# ── SEARCH / SCORING ENGINE ───────────────────────────────────────────────────

@dataclass(frozen=True)
class SpotifyCandidate:
    track_uri: str
    track_name: str
    artist_name: str
    score: float
    source: str
    popularity: int = 0
    album_name: str = ""


def _combined_aliases(ctx: ToolContext) -> dict[str, str]:
    aliases: dict[str, str] = {}
    aliases.update(ctx.config.transcript_aliases or {})
    aliases.update(ctx.config.spotify_aliases or {})
    return aliases


def _clean_query(raw_query: str, aliases: dict[str, str]) -> str:
    query = normalize_text(raw_query, aliases=aliases)
    return _RE_COMMAND_PREFIX.sub("", query).strip()


def _split_query(query: str) -> tuple[str, str | None]:
    for sep in (" by ", " from ", " - ", " of "):
        if sep in query:
            left, right = query.split(sep, 1)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return query, None


def _track_text(track: dict[str, Any]) -> tuple[str, str]:
    name = track.get("name", "")
    artists = ", ".join(a.get("name", "") for a in track.get("artists", []))
    return name, artists


def _score_track(
    *,
    track_name: str,
    artist_name: str,
    popularity: int,
    raw_query: str,
    title_query: str,
    artist_query: str | None,
    source: str,
) -> float:
    title_base = title_query or raw_query
    artist_base = artist_query or raw_query
    combined = f"{track_name} {artist_name}".strip()

    title_sim = max(similarity(track_name, title_base), token_overlap(track_name, title_base))
    artist_sim = max(similarity(artist_name, artist_base), token_overlap(artist_name, artist_base))
    query_sim = max(similarity(combined, raw_query), token_overlap(combined, raw_query))

    score = (
        (0.50 * title_sim) + (0.40 * artist_sim) + (0.10 * query_sim)
        if artist_query
        else (0.60 * title_sim) + (0.20 * artist_sim) + (0.20 * query_sim)
    )
    lowered = combined.casefold()
    if any(t in lowered for t in (" live", " remix", " karaoke", " cover")) and not any(
        t in raw_query.casefold() for t in ("live", "remix", "karaoke", "cover")
    ):
        score -= 0.08
    if title_query and title_query.casefold() in track_name.casefold():
        score += 0.07
    if artist_query and artist_query.casefold() in artist_name.casefold():
        score += 0.10
    score += min(max(popularity, 0), 100) / 1000.0
    return max(0.0, min(score, 1.0))


def _collect_track_candidates(
    sp: Any, *, query: str, title_query: str, artist_query: str | None,
    limit: int, market: str,
) -> list[SpotifyCandidate]:
    search_queries = []
    if title_query and artist_query:
        search_queries.extend([
            f'track:"{title_query}" artist:"{artist_query}"',
            f"track:{title_query} artist:{artist_query}",
            f"{title_query} {artist_query}",
        ])
    search_queries.extend([query, title_query or query])
    if artist_query:
        search_queries.append(artist_query)

    candidates: list[SpotifyCandidate] = []
    seen: set[str] = set()
    for sq in dict.fromkeys(search_queries):
        try:
            result = sp.search(q=sq, type="track", limit=limit, market=market)
        except Exception as e:
            logger.debug("track search failed q=%s err=%s", sq, e)
            continue
        for track in result.get("tracks", {}).get("items", []):
            track_uri = track.get("uri")
            track_name, artist_name = _track_text(track)
            if not track_uri or not track_name:
                continue
            key = f"{track_uri}:{sq}"
            if key in seen:
                continue
            seen.add(key)
            candidates.append(SpotifyCandidate(
                track_uri=track_uri, track_name=track_name, artist_name=artist_name,
                score=0.0, source=f"track:{sq}",
                popularity=int(track.get("popularity") or 0),
                album_name=track.get("album", {}).get("name", ""),
            ))
    return candidates


def _collect_artist_candidates(
    sp: Any, *, query: str, artist_query: str, limit: int, market: str,
) -> list[SpotifyCandidate]:
    candidates: list[SpotifyCandidate] = []
    try:
        result = sp.search(q=artist_query or query, type="artist", limit=min(limit, 5))
    except Exception as e:
        logger.debug("artist search failed q=%s err=%s", artist_query or query, e)
        return candidates

    for artist in result.get("artists", {}).get("items", [])[:3]:
        artist_id = artist.get("id")
        artist_name = artist.get("name", "")
        if not artist_id or not artist_name:
            continue
        try:
            # Use search instead of artist_top_tracks to avoid the country=US 403 bug
            tr = sp.search(q=f'artist:"{artist_name}"', type="track", limit=3, market=market)
            top_tracks = tr.get("tracks", {}).get("items", [])
        except Exception as e:
            logger.debug("artist track search failed artist=%s err=%s", artist_name, e)
            continue
        for track in top_tracks[:3]:
            track_uri = track.get("uri")
            track_name, artists = _track_text(track)
            if not track_uri or not track_name:
                continue
            candidates.append(SpotifyCandidate(
                track_uri=track_uri, track_name=track_name, artist_name=artists or artist_name,
                score=0.0, source=f"artist:{artist_name}",
                popularity=int(track.get("popularity") or 0),
                album_name=track.get("album", {}).get("name", ""),
            ))
    return candidates


def _rank_candidates(
    candidates: list[SpotifyCandidate], *, raw_query: str,
    title_query: str, artist_query: str | None,
) -> list[SpotifyCandidate]:
    ranked = []
    for c in candidates:
        score = _score_track(
            track_name=c.track_name, artist_name=c.artist_name, popularity=c.popularity,
            raw_query=raw_query, title_query=title_query, artist_query=artist_query, source=c.source,
        )
        ranked.append(SpotifyCandidate(
            track_uri=c.track_uri, track_name=c.track_name, artist_name=c.artist_name,
            score=score, source=c.source, popularity=c.popularity, album_name=c.album_name,
        ))
    ranked.sort(key=lambda x: (x.score, x.popularity), reverse=True)
    return ranked


def _select_best_candidate(sp: Any, ctx: ToolContext, raw_query: str) -> dict[str, Any]:
    market = _get_market(sp)
    aliases = _combined_aliases(ctx)
    cleaned = _clean_query(raw_query, aliases)
    title_query, artist_query = _split_query(cleaned)
    search_limit = max(1, min(int(ctx.config.spotify_search_limit), 25))

    candidates = _collect_track_candidates(
        sp, query=cleaned, title_query=title_query,
        artist_query=artist_query, limit=search_limit, market=market,
    )
    candidates.extend(_collect_artist_candidates(
        sp, query=cleaned, artist_query=artist_query or cleaned,
        limit=search_limit, market=market,
    ))
    ranked = _rank_candidates(candidates, raw_query=cleaned, title_query=title_query, artist_query=artist_query)

    if not ranked:
        return {"success": False, "error": f"No tracks found matching: '{raw_query}'"}

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    confidence = best.score
    gap = confidence - (runner_up.score if runner_up else 0.0)

    if (confidence < ctx.config.spotify_clarify_confidence or
            (runner_up and confidence < ctx.config.spotify_accept_confidence and gap < 0.12)):
        return {
            "success": True, "needs_clarification": True, "confidence": confidence,
            "question": f"Did you mean '{best.track_name}' by {best.artist_name}?",
            "suggestion": {"track_name": best.track_name, "artist": best.artist_name, "album": best.album_name},
        }

    return {
        "success": True,
        "status": f"Now playing: '{best.track_name}' by {best.artist_name}",
        "track_name": best.track_name, "artist": best.artist_name, "album": best.album_name,
        "confidence": confidence, "matched_from": best.source,
        "selected_track_uri": best.track_uri,
        "selected_track_id": best.track_uri.split(":")[-1],
    }


def _start_playback(sp: Any, track_uri: str | None = None, context_uri: str | None = None) -> None:
    """Smart playback start — auto-selects device, falls back to first available."""
    devices = sp.devices().get("devices", [])
    if not devices:
        raise RuntimeError(
            "No active Spotify device found. Please open Spotify on your desktop or phone first."
        )
    active = next((d for d in devices if d.get("is_active")), None)
    device_id = active["id"] if active else devices[0]["id"]
    kwargs: dict[str, Any] = {"device_id": device_id}
    if track_uri:
        kwargs["uris"] = [track_uri]
    if context_uri:
        kwargs["context_uri"] = context_uri
    sp.start_playback(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — PLAYER TOOLS
# ══════════════════════════════════════════════════════════════════════════════

class SpotifyPlayArgs(BaseModel):
    query: str = Field(description="Song name, 'song by artist', album name, or playlist name.")

class SpotifyPauseArgs(BaseModel):
    pass

class SpotifyResumeArgs(BaseModel):
    pass

class SpotifySkipNextArgs(BaseModel):
    pass

class SpotifySkipPreviousArgs(BaseModel):
    pass

class SpotifyStateArgs(BaseModel):
    pass

class SpotifySetVolumeArgs(BaseModel):
    percent: int = Field(description="Volume level 0–100.")

class SpotifySetShuffleArgs(BaseModel):
    enabled: bool = Field(description="True to enable shuffle, False to disable.")

class SpotifySetRepeatArgs(BaseModel):
    mode: Literal["off", "track", "context"] = Field(
        description="'off', 'track' (loop one song), or 'context' (loop album/playlist)."
    )

class SpotifySeekArgs(BaseModel):
    position_seconds: int = Field(description="Seconds from start to seek to.")

class SpotifyQueueArgs(BaseModel):
    query: str = Field(description="Song name to search for and add to the play queue.")

class SpotifyDevicesArgs(BaseModel):
    pass

class SpotifyTransferArgs(BaseModel):
    device_name: str = Field(description="Name or partial name of the target Spotify device.")

class SpotifyRecentArgs(BaseModel):
    limit: int = Field(default=10, description="Number of recently played tracks to return (max 50).")


def handle_spotify_play(ctx: ToolContext, args: SpotifyPlayArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        selection = _select_best_candidate(sp, ctx, args.query)
        if not selection.get("success") or selection.get("needs_clarification"):
            return selection
        track_uri = selection.pop("selected_track_uri")
        selection.pop("selected_track_id", None)
        _start_playback(sp, track_uri=track_uri)
        return selection
    except (PermissionError, RuntimeError) as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error("Spotify play failed: %s", e)
        return {"success": False, "error": str(e)}


def handle_spotify_pause(ctx: ToolContext, args: SpotifyPauseArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.pause_playback()
        return {"success": True, "status": "Spotify playback paused."}
    except Exception as e:
        if "Restriction violated" in str(e):
            return {"success": True, "status": "Spotify is already paused."}
        return {"success": False, "error": str(e)}


def handle_spotify_resume(ctx: ToolContext, args: SpotifyResumeArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.start_playback()
        return {"success": True, "status": "Spotify playback resumed."}
    except Exception as e:
        if "Restriction violated" in str(e):
            return {"success": True, "status": "Spotify is already playing."}
        return {"success": False, "error": str(e)}


def handle_spotify_skip_next(ctx: ToolContext, args: SpotifySkipNextArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.next_track()
        return {"success": True, "status": "Skipped to next track."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_skip_previous(ctx: ToolContext, args: SpotifySkipPreviousArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.previous_track()
        return {"success": True, "status": "Went back to previous track."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_state(ctx: ToolContext, args: SpotifyStateArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        state = sp.current_playback(market=_get_market(sp))
        if not state:
            return {"success": True, "is_playing": False, "status": "Nothing is currently playing on Spotify."}
        item = state.get("item") or {}
        device = state.get("device") or {}
        return {
            "success": True,
            "is_playing": state.get("is_playing", False),
            "track_name": item.get("name", "Unknown"),
            "artist": ", ".join(a["name"] for a in item.get("artists", [])),
            "album": item.get("album", {}).get("name", "Unknown"),
            "progress_seconds": (state.get("progress_ms") or 0) // 1000,
            "duration_seconds": (item.get("duration_ms") or 0) // 1000,
            "volume": device.get("volume_percent"),
            "shuffle": state.get("shuffle_state"),
            "repeat": state.get("repeat_state"),
            "device_name": device.get("name"),
            "track_uri": item.get("uri"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_set_volume(ctx: ToolContext, args: SpotifySetVolumeArgs) -> dict[str, Any]:
    try:
        percent = max(0, min(100, args.percent))
        sp = _get_spotify_client(ctx)
        sp.volume(percent)
        return {"success": True, "status": f"Volume set to {percent}%."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_set_shuffle(ctx: ToolContext, args: SpotifySetShuffleArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.shuffle(args.enabled)
        return {"success": True, "status": f"Shuffle {'enabled' if args.enabled else 'disabled'}."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_set_repeat(ctx: ToolContext, args: SpotifySetRepeatArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.repeat(args.mode)
        labels = {"off": "off", "track": "repeating current track", "context": "repeating playlist/album"}
        return {"success": True, "status": f"Repeat set to: {labels.get(args.mode, args.mode)}."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_seek(ctx: ToolContext, args: SpotifySeekArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.seek_track(max(0, args.position_seconds) * 1000)
        return {"success": True, "status": f"Seeked to {args.position_seconds}s."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_queue(ctx: ToolContext, args: SpotifyQueueArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        selection = _select_best_candidate(sp, ctx, args.query)
        if not selection.get("success") or selection.get("needs_clarification"):
            return selection
        sp.add_to_queue(selection["selected_track_uri"])
        return {"success": True, "status": f"Added '{selection['track_name']}' by {selection['artist']} to queue."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_devices(ctx: ToolContext, args: SpotifyDevicesArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        devices = [
            {"id": d["id"], "name": d["name"], "type": d["type"],
             "is_active": d["is_active"], "volume": d.get("volume_percent")}
            for d in sp.devices().get("devices", [])
        ]
        return {"success": True, "devices": devices, "count": len(devices)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_transfer(ctx: ToolContext, args: SpotifyTransferArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        device_list = sp.devices().get("devices", [])
        target = next((d for d in device_list if args.device_name.lower() in d["name"].lower()), None)
        if not target:
            return {"success": False, "error": f"Device '{args.device_name}' not found. Available: {[d['name'] for d in device_list]}"}
        sp.transfer_playback(target["id"], force_play=True)
        return {"success": True, "status": f"Playback transferred to '{target['name']}'."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_spotify_recent(ctx: ToolContext, args: SpotifyRecentArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        result = sp.current_user_recently_played(limit=max(1, min(args.limit, 50)))
        tracks = [
            {"track_name": i["track"]["name"], "artist": i["track"]["artists"][0]["name"],
             "track_uri": i["track"]["uri"], "played_at": i["played_at"]}
            for i in result.get("items", [])
        ]
        return {"success": True, "tracks": tracks, "count": len(tracks)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------------------
# SECTION B � PLAYLIST TOOLS
# ------------------------------------------------------------------------------

class PlaylistListArgs(BaseModel):
    pass

class PlaylistGetArgs(BaseModel):
    playlist_id: str = Field(description="Spotify playlist ID to fetch tracks from.")

class PlaylistCreateArgs(BaseModel):
    name: str = Field(description="Name for the new playlist.")
    description: str = Field(default="", description="Optional description.")
    public: bool = Field(default=False, description="True for public, False for private.")

class PlaylistAddTracksArgs(BaseModel):
    playlist_id: str = Field(description="Spotify playlist ID.")
    queries: list[str] = Field(description="List of song names/queries to search and add.")

class PlaylistRemoveTracksArgs(BaseModel):
    playlist_id: str = Field(description="Spotify playlist ID.")
    track_uris: list[str] = Field(description="List of Spotify track URIs to remove.")

class PlaylistReorderArgs(BaseModel):
    playlist_id: str = Field(description="Spotify playlist ID.")
    range_start: int = Field(description="0-based index of track to move.")
    insert_before: int = Field(description="Position index to insert before.")
    range_length: int = Field(default=1, description="Number of consecutive tracks to move.")

class PlaylistUpdateArgs(BaseModel):
    playlist_id: str = Field(description="Spotify playlist ID.")
    name: str | None = Field(default=None, description="New name for the playlist.")
    description: str | None = Field(default=None, description="New description.")
    public: bool | None = Field(default=None, description="Change public/private state.")


def handle_playlist_list(ctx: ToolContext, args: PlaylistListArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        playlists, offset = [], 0
        while True:
            result = sp.current_user_playlists(limit=50, offset=offset)
            items = result.get("items", [])
            playlists.extend([
                {"id": p["id"], "name": p["name"],
                 "track_count": p["tracks"]["total"], "public": p["public"],
                 "owner": p["owner"]["display_name"]}
                for p in items if p
            ])
            if not result.get("next"):
                break
            offset += 50
        return {"success": True, "playlists": playlists, "count": len(playlists)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_playlist_get(ctx: ToolContext, args: PlaylistGetArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        market = _get_market(sp)
        tracks, offset = [], 0
        while True:
            result = sp.playlist_items(
                args.playlist_id, market=market, limit=100, offset=offset,
                fields="items(track(id,name,uri,artists,album)),next",
            )
            for item in result.get("items", []):
                track = item.get("track")
                if track:
                    tracks.append({
                        "name": track["name"],
                        "artist": track["artists"][0]["name"],
                        "uri": track["uri"], "id": track["id"],
                    })
            if not result.get("next"):
                break
            offset += 100
        meta = sp.playlist(args.playlist_id)
        return {
            "success": True, "name": meta["name"],
            "description": meta.get("description", ""),
            "tracks": tracks, "total": len(tracks),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_playlist_create(ctx: ToolContext, args: PlaylistCreateArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        user_id = sp.current_user()["id"]
        playlist = sp.user_playlist_create(
            user=user_id, name=args.name, public=args.public,
            collaborative=False, description=args.description,
        )
        return {"success": True, "playlist_id": playlist["id"],
                "status": f"Playlist '{args.name}' created successfully."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_playlist_add_tracks(ctx: ToolContext, args: PlaylistAddTracksArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        uris, not_found = [], []
        for query in args.queries:
            sel = _select_best_candidate(sp, ctx, query)
            if sel.get("success") and not sel.get("needs_clarification"):
                uris.append(sel["selected_track_uri"])
            else:
                not_found.append(query)
        if not uris:
            return {"success": False, "error": f"No tracks found for: {not_found}"}
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(args.playlist_id, uris[i:i+100])
        return {"success": True, "added": len(uris), "not_found": not_found,
                "status": f"Added {len(uris)} track(s) to playlist."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_playlist_remove_tracks(ctx: ToolContext, args: PlaylistRemoveTracksArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        for i in range(0, len(args.track_uris), 100):
            sp.playlist_remove_all_occurrences_of_items(args.playlist_id, args.track_uris[i:i+100])
        return {"success": True, "status": f"Removed {len(args.track_uris)} track(s) from playlist."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_playlist_reorder(ctx: ToolContext, args: PlaylistReorderArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        result = sp.playlist_reorder_items(
            args.playlist_id, range_start=args.range_start,
            insert_before=args.insert_before, range_length=args.range_length,
        )
        return {"success": True, "snapshot_id": result["snapshot_id"],
                "status": "Playlist tracks reordered."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_playlist_update(ctx: ToolContext, args: PlaylistUpdateArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        kwargs: dict[str, Any] = {}
        if args.name is not None:
            kwargs["name"] = args.name
        if args.description is not None:
            kwargs["description"] = args.description
        if args.public is not None:
            kwargs["public"] = args.public
        if not kwargs:
            return {"success": False, "error": "No fields provided to update."}
        sp.playlist_change_details(args.playlist_id, **kwargs)
        return {"success": True, "status": "Playlist details updated."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------------------
# SECTION C � LIBRARY TOOLS
# ------------------------------------------------------------------------------

class LibraryLikeArgs(BaseModel):
    query: str | None = Field(default=None, description="Song name to like. If omitted, likes the currently playing track.")

class LibraryUnlikeArgs(BaseModel):
    query: str = Field(description="Song name to remove from Liked Songs.")

class LibraryLikedTracksArgs(BaseModel):
    limit: int = Field(default=20, description="Number of liked songs to return (max 50).")
    offset: int = Field(default=0, description="Offset for pagination.")

class LibrarySaveAlbumArgs(BaseModel):
    query: str = Field(description="Album name (optionally 'album by artist') to save.")

class LibrarySavedAlbumsArgs(BaseModel):
    limit: int = Field(default=20, description="Number of saved albums to return (max 50).")


def handle_library_like(ctx: ToolContext, args: LibraryLikeArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        if args.query:
            sel = _select_best_candidate(sp, ctx, args.query)
            if not sel.get("success") or sel.get("needs_clarification"):
                return sel
            track_id = sel["selected_track_id"]
            track_name = sel["track_name"]
            artist = sel["artist"]
        else:
            state = sp.current_playback()
            if not state or not state.get("item"):
                return {"success": False, "error": "Nothing is currently playing."}
            track_id = state["item"]["id"]
            track_name = state["item"]["name"]
            artist = state["item"]["artists"][0]["name"]
        sp.current_user_saved_tracks_add([track_id])
        return {"success": True, "status": f"Liked '{track_name}' by {artist}."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_library_unlike(ctx: ToolContext, args: LibraryUnlikeArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sel = _select_best_candidate(sp, ctx, args.query)
        if not sel.get("success") or sel.get("needs_clarification"):
            return sel
        sp.current_user_saved_tracks_delete([sel["selected_track_id"]])
        return {"success": True, "status": f"Removed '{sel['track_name']}' by {sel['artist']} from Liked Songs."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_library_liked_tracks(ctx: ToolContext, args: LibraryLikedTracksArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        result = sp.current_user_saved_tracks(
            limit=max(1, min(args.limit, 50)), offset=args.offset,
            market=_get_market(sp),
        )
        tracks = [
            {"name": i["track"]["name"], "artist": i["track"]["artists"][0]["name"],
             "uri": i["track"]["uri"], "added_at": i["added_at"]}
            for i in result.get("items", [])
        ]
        return {"success": True, "tracks": tracks, "total": result.get("total", 0)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_library_save_album(ctx: ToolContext, args: LibrarySaveAlbumArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        market = _get_market(sp)
        result = sp.search(q=args.query, type="album", limit=1, market=market)
        albums = result.get("albums", {}).get("items", [])
        if not albums:
            return {"success": False, "error": f"No album found matching: '{args.query}'"}
        album = albums[0]
        sp.current_user_saved_albums_add([album["id"]])
        return {"success": True, "status": f"Saved album '{album['name']}' by {album['artists'][0]['name']}."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_library_saved_albums(ctx: ToolContext, args: LibrarySavedAlbumsArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        result = sp.current_user_saved_albums(
            limit=max(1, min(args.limit, 50)), offset=0, market=_get_market(sp),
        )
        albums = [
            {"name": i["album"]["name"], "artist": i["album"]["artists"][0]["name"],
             "id": i["album"]["id"], "added_at": i["added_at"]}
            for i in result.get("items", [])
        ]
        return {"success": True, "albums": albums, "total": result.get("total", 0)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------------------
# SECTION D � PERSONALIZATION TOOLS
# ------------------------------------------------------------------------------

class TopTracksArgs(BaseModel):
    limit: int = Field(default=10, description="Number of top tracks to return (max 50).")
    time_range: Literal["short_term", "medium_term", "long_term"] = Field(
        default="medium_term",
        description="'short_term' (last 4 weeks), 'medium_term' (6 months), 'long_term' (all time).",
    )

class TopArtistsArgs(BaseModel):
    limit: int = Field(default=10, description="Number of top artists to return (max 50).")
    time_range: Literal["short_term", "medium_term", "long_term"] = Field(
        default="medium_term",
        description="'short_term' (last 4 weeks), 'medium_term' (6 months), 'long_term' (all time).",
    )


def handle_top_tracks(ctx: ToolContext, args: TopTracksArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        result = sp.current_user_top_tracks(
            limit=max(1, min(args.limit, 50)), time_range=args.time_range,
        )
        tracks = [
            {"name": t["name"], "artist": t["artists"][0]["name"],
             "uri": t["uri"], "id": t["id"], "popularity": t["popularity"]}
            for t in result.get("items", [])
        ]
        return {"success": True, "tracks": tracks, "time_range": args.time_range}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_top_artists(ctx: ToolContext, args: TopArtistsArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        result = sp.current_user_top_artists(
            limit=max(1, min(args.limit, 50)), time_range=args.time_range,
        )
        artists = [
            {"name": a["name"], "id": a["id"], "genres": a["genres"], "popularity": a["popularity"]}
            for a in result.get("items", [])
        ]
        return {"success": True, "artists": artists, "time_range": args.time_range}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------------------
# SECTION E � CATALOG TOOLS
# ------------------------------------------------------------------------------

class CatalogArtistArgs(BaseModel):
    artist_name: str = Field(description="Artist name to look up.")

class CatalogArtistAlbumsArgs(BaseModel):
    artist_name: str = Field(description="Artist name.")
    include_groups: str = Field(default="album,single", description="Comma-separated: album, single, appears_on, compilation.")

class CatalogArtistTopTracksArgs(BaseModel):
    artist_name: str = Field(description="Artist name to get top tracks for.")

class CatalogAlbumArgs(BaseModel):
    query: str = Field(description="Album name (optionally 'album by artist').")


def _search_artist(sp: Any, name: str) -> dict | None:
    result = sp.search(q=name, type="artist", limit=1)
    items = result.get("artists", {}).get("items", [])
    return items[0] if items else None


def handle_catalog_artist(ctx: ToolContext, args: CatalogArtistArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        aliases = _combined_aliases(ctx)
        name = normalize_text(args.artist_name, aliases=aliases)
        artist = _search_artist(sp, name)
        if not artist:
            return {"success": False, "error": f"Artist '{args.artist_name}' not found."}
        return {
            "success": True, "name": artist["name"], "id": artist["id"],
            "genres": artist["genres"], "popularity": artist["popularity"],
            "followers": artist["followers"]["total"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_catalog_artist_albums(ctx: ToolContext, args: CatalogArtistAlbumsArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        market = _get_market(sp)
        aliases = _combined_aliases(ctx)
        name = normalize_text(args.artist_name, aliases=aliases)
        artist = _search_artist(sp, name)
        if not artist:
            return {"success": False, "error": f"Artist '{args.artist_name}' not found."}
        result = sp.artist_albums(
            artist["id"], album_type=args.include_groups, country=market, limit=50,
        )
        albums = [{"name": a["name"], "id": a["id"], "type": a["album_type"],
                   "release_date": a["release_date"]} for a in result.get("items", [])]
        while result.get("next"):
            result = sp.next(result)
            albums.extend([{"name": a["name"], "id": a["id"], "type": a["album_type"],
                            "release_date": a["release_date"]} for a in result.get("items", [])])
        return {"success": True, "artist": artist["name"], "albums": albums, "count": len(albums)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_catalog_artist_top_tracks(ctx: ToolContext, args: CatalogArtistTopTracksArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        market = _get_market(sp)
        aliases = _combined_aliases(ctx)
        name = normalize_text(args.artist_name, aliases=aliases)
        artist = _search_artist(sp, name)
        if not artist:
            return {"success": False, "error": f"Artist '{args.artist_name}' not found."}
        # Use artist_top_tracks correctly with the user's own market � no more country=US 403
        result = sp.artist_top_tracks(artist["id"], country=market)
        tracks = [
            {"name": t["name"], "uri": t["uri"], "id": t["id"],
             "album": t["album"]["name"], "popularity": t["popularity"]}
            for t in result.get("tracks", [])
        ]
        return {"success": True, "artist": artist["name"], "tracks": tracks}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_catalog_album(ctx: ToolContext, args: CatalogAlbumArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        market = _get_market(sp)
        result = sp.search(q=args.query, type="album", limit=1, market=market)
        albums = result.get("albums", {}).get("items", [])
        if not albums:
            return {"success": False, "error": f"No album found matching: '{args.query}'"}
        alb = albums[0]
        full = sp.album(alb["id"], market=market)
        tracks = [
            {"name": t["name"], "uri": t["uri"], "track_number": t["track_number"]}
            for t in full.get("tracks", {}).get("items", [])
        ]
        return {
            "success": True, "name": full["name"], "id": full["id"],
            "artist": full["artists"][0]["name"], "release_date": full["release_date"],
            "total_tracks": full["total_tracks"], "tracks": tracks,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------------------
# SECTION F � SOCIAL TOOLS
# ------------------------------------------------------------------------------

class FollowArtistArgs(BaseModel):
    artist_name: str = Field(description="Artist name to follow on Spotify.")

class UnfollowArtistArgs(BaseModel):
    artist_name: str = Field(description="Artist name to unfollow on Spotify.")


def handle_follow_artist(ctx: ToolContext, args: FollowArtistArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        aliases = _combined_aliases(ctx)
        name = normalize_text(args.artist_name, aliases=aliases)
        artist = _search_artist(sp, name)
        if not artist:
            return {"success": False, "error": f"Artist '{args.artist_name}' not found."}
        sp.user_follow_artists([artist["id"]])
        return {"success": True, "status": f"Now following {artist['name']}."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_unfollow_artist(ctx: ToolContext, args: UnfollowArtistArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        aliases = _combined_aliases(ctx)
        name = normalize_text(args.artist_name, aliases=aliases)
        artist = _search_artist(sp, name)
        if not artist:
            return {"success": False, "error": f"Artist '{args.artist_name}' not found."}
        sp.user_unfollow_artists([artist["id"]])
        return {"success": True, "status": f"Unfollowed {artist['name']}."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------------------
# TOOL REGISTRY � build()
# All 22 tools registered with descriptions optimised for the LLM to pick correctly.
# ------------------------------------------------------------------------------

def build() -> list[Tool]:
    return [
        # -- PLAYER ----------------------------------------------------------
        Tool[SpotifyPlayArgs](
            name="spotify.play",
            description=(
                "Search for a song, artist, album, or playlist on Spotify and play it immediately. "
                "Accepts queries like 'London by Money Aujla', 'Sidhu Moosewala songs', or just a song name. "
                "Requires Spotify to be open on a device. Requires Spotify Premium."
            ),
            input_model=SpotifyPlayArgs,
            handler=handle_spotify_play,
            side_effect=True,
        ),
        Tool[SpotifyPauseArgs](
            name="spotify.pause",
            description="Pause the current Spotify playback. Requires Spotify Premium.",
            input_model=SpotifyPauseArgs,
            handler=handle_spotify_pause,
            side_effect=True,
        ),
        Tool[SpotifyResumeArgs](
            name="spotify.resume",
            description="Resume paused Spotify playback. Requires Spotify Premium.",
            input_model=SpotifyResumeArgs,
            handler=handle_spotify_resume,
            side_effect=True,
        ),
        Tool[SpotifySkipNextArgs](
            name="spotify.skip_next",
            description="Skip to the next track on Spotify. Requires Spotify Premium.",
            input_model=SpotifySkipNextArgs,
            handler=handle_spotify_skip_next,
            side_effect=True,
        ),
        Tool[SpotifySkipPreviousArgs](
            name="spotify.skip_previous",
            description="Go back to the previous track on Spotify. Requires Spotify Premium.",
            input_model=SpotifySkipPreviousArgs,
            handler=handle_spotify_skip_previous,
            side_effect=True,
        ),
        Tool[SpotifyStateArgs](
            name="spotify.state",
            description=(
                "Get the current Spotify playback state: what song is playing, artist, album, "
                "progress, volume, shuffle, repeat mode, and device name."
            ),
            input_model=SpotifyStateArgs,
            handler=handle_spotify_state,
            side_effect=False,
        ),
        Tool[SpotifySetVolumeArgs](
            name="spotify.set_volume",
            description="Set the Spotify playback volume to a specific level (0-100). Requires Spotify Premium.",
            input_model=SpotifySetVolumeArgs,
            handler=handle_spotify_set_volume,
            side_effect=True,
        ),
        Tool[SpotifySetShuffleArgs](
            name="spotify.set_shuffle",
            description="Enable or disable shuffle mode on Spotify. Requires Spotify Premium.",
            input_model=SpotifySetShuffleArgs,
            handler=handle_spotify_set_shuffle,
            side_effect=True,
        ),
        Tool[SpotifySetRepeatArgs](
            name="spotify.set_repeat",
            description=(
                "Set repeat mode on Spotify: 'off', 'track' (loop current song), "
                "or 'context' (loop current album/playlist). Requires Spotify Premium."
            ),
            input_model=SpotifySetRepeatArgs,
            handler=handle_spotify_set_repeat,
            side_effect=True,
        ),
        Tool[SpotifySeekArgs](
            name="spotify.seek",
            description="Jump to a specific position (in seconds) in the currently playing track. Requires Spotify Premium.",
            input_model=SpotifySeekArgs,
            handler=handle_spotify_seek,
            side_effect=True,
        ),
        Tool[SpotifyQueueArgs](
            name="spotify.queue",
            description="Search for a song and add it to the Spotify play queue. Requires Spotify Premium.",
            input_model=SpotifyQueueArgs,
            handler=handle_spotify_queue,
            side_effect=True,
        ),
        Tool[SpotifyDevicesArgs](
            name="spotify.devices",
            description="List all available Spotify devices (desktop app, phone, smart speaker, etc.) for this account.",
            input_model=SpotifyDevicesArgs,
            handler=handle_spotify_devices,
            side_effect=False,
        ),
        Tool[SpotifyTransferArgs](
            name="spotify.transfer",
            description="Transfer Spotify playback to a different device by name. Requires Spotify Premium.",
            input_model=SpotifyTransferArgs,
            handler=handle_spotify_transfer,
            side_effect=True,
        ),
        Tool[SpotifyRecentArgs](
            name="spotify.recent",
            description="Get the user's recently played Spotify tracks in reverse chronological order.",
            input_model=SpotifyRecentArgs,
            handler=handle_spotify_recent,
            side_effect=False,
        ),
        # -- PLAYLISTS --------------------------------------------------------
        Tool[PlaylistListArgs](
            name="spotify.playlist.list",
            description="List all playlists owned or followed by the user on Spotify.",
            input_model=PlaylistListArgs,
            handler=handle_playlist_list,
            side_effect=False,
        ),
        Tool[PlaylistGetArgs](
            name="spotify.playlist.get",
            description="Get all tracks inside a specific Spotify playlist by its ID.",
            input_model=PlaylistGetArgs,
            handler=handle_playlist_get,
            side_effect=False,
        ),
        Tool[PlaylistCreateArgs](
            name="spotify.playlist.create",
            description="Create a new Spotify playlist for the user with a given name, description, and public/private setting.",
            input_model=PlaylistCreateArgs,
            handler=handle_playlist_create,
            side_effect=True,
        ),
        Tool[PlaylistAddTracksArgs](
            name="spotify.playlist.add_tracks",
            description="Search for songs by name and add them to a Spotify playlist.",
            input_model=PlaylistAddTracksArgs,
            handler=handle_playlist_add_tracks,
            side_effect=True,
        ),
        Tool[PlaylistRemoveTracksArgs](
            name="spotify.playlist.remove_tracks",
            description="Remove specific tracks (by URI) from a Spotify playlist.",
            input_model=PlaylistRemoveTracksArgs,
            handler=handle_playlist_remove_tracks,
            side_effect=True,
        ),
        Tool[PlaylistReorderArgs](
            name="spotify.playlist.reorder",
            description="Move tracks to a different position inside a Spotify playlist.",
            input_model=PlaylistReorderArgs,
            handler=handle_playlist_reorder,
            side_effect=True,
        ),
        Tool[PlaylistUpdateArgs](
            name="spotify.playlist.update",
            description="Update a Spotify playlist's name, description, or public/private status.",
            input_model=PlaylistUpdateArgs,
            handler=handle_playlist_update,
            side_effect=True,
        ),
        # -- LIBRARY ----------------------------------------------------------
        Tool[LibraryLikeArgs](
            name="spotify.library.like",
            description=(
                "Like (save) a song to the user's Liked Songs library. "
                "If a song name is given, searches for it. If no name given, likes the currently playing track."
            ),
            input_model=LibraryLikeArgs,
            handler=handle_library_like,
            side_effect=True,
        ),
        Tool[LibraryUnlikeArgs](
            name="spotify.library.unlike",
            description="Remove a song from the user's Liked Songs library.",
            input_model=LibraryUnlikeArgs,
            handler=handle_library_unlike,
            side_effect=True,
        ),
        Tool[LibraryLikedTracksArgs](
            name="spotify.library.liked_tracks",
            description="Get the user's Liked Songs library, most recently added first.",
            input_model=LibraryLikedTracksArgs,
            handler=handle_library_liked_tracks,
            side_effect=False,
        ),
        Tool[LibrarySaveAlbumArgs](
            name="spotify.library.save_album",
            description="Search for an album and save it to the user's Spotify library.",
            input_model=LibrarySaveAlbumArgs,
            handler=handle_library_save_album,
            side_effect=True,
        ),
        Tool[LibrarySavedAlbumsArgs](
            name="spotify.library.saved_albums",
            description="Get the list of albums saved in the user's Spotify library.",
            input_model=LibrarySavedAlbumsArgs,
            handler=handle_library_saved_albums,
            side_effect=False,
        ),
        # -- PERSONALIZATION --------------------------------------------------
        Tool[TopTracksArgs](
            name="spotify.top_tracks",
            description=(
                "Get the user's most listened-to tracks. "
                "time_range: 'short_term' (last month), 'medium_term' (6 months), 'long_term' (all time)."
            ),
            input_model=TopTracksArgs,
            handler=handle_top_tracks,
            side_effect=False,
        ),
        Tool[TopArtistsArgs](
            name="spotify.top_artists",
            description=(
                "Get the user's most listened-to artists including their genres. "
                "Useful for taste profiling without the restricted /recommendations endpoint."
            ),
            input_model=TopArtistsArgs,
            handler=handle_top_artists,
            side_effect=False,
        ),
        # -- CATALOG ----------------------------------------------------------
        Tool[CatalogArtistArgs](
            name="spotify.catalog.artist",
            description="Look up an artist's profile on Spotify: genres, popularity, follower count.",
            input_model=CatalogArtistArgs,
            handler=handle_catalog_artist,
            side_effect=False,
        ),
        Tool[CatalogArtistAlbumsArgs](
            name="spotify.catalog.artist_albums",
            description="Get all albums and singles released by a specific artist on Spotify.",
            input_model=CatalogArtistAlbumsArgs,
            handler=handle_catalog_artist_albums,
            side_effect=False,
        ),
        Tool[CatalogArtistTopTracksArgs](
            name="spotify.catalog.artist_top_tracks",
            description=(
                "Get an artist's top tracks in the user's country (market-aware, works for Indian/Punjabi artists). "
                "Use this instead of spotify.play when the user asks 'what are Sidhu's best songs' etc."
            ),
            input_model=CatalogArtistTopTracksArgs,
            handler=handle_catalog_artist_top_tracks,
            side_effect=False,
        ),
        Tool[CatalogAlbumArgs](
            name="spotify.catalog.album",
            description="Get full details and track listing for a specific album on Spotify.",
            input_model=CatalogAlbumArgs,
            handler=handle_catalog_album,
            side_effect=False,
        ),
        # -- SOCIAL -----------------------------------------------------------
        Tool[FollowArtistArgs](
            name="spotify.follow_artist",
            description="Follow an artist on Spotify so they appear in the user's library and new release feed.",
            input_model=FollowArtistArgs,
            handler=handle_follow_artist,
            side_effect=True,
        ),
        Tool[UnfollowArtistArgs](
            name="spotify.unfollow_artist",
            description="Unfollow an artist on Spotify.",
            input_model=UnfollowArtistArgs,
            handler=handle_unfollow_artist,
            side_effect=True,
        ),
    ]
