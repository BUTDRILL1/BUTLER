from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext
from butler.voice.normalize import normalize_text, similarity, token_overlap

logger = logging.getLogger(__name__)


_RE_COMMAND_PREFIX = re.compile(r"^\s*(?:play|put on|start|start playing|listen to)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class SpotifyCandidate:
    track_uri: str
    track_name: str
    artist_name: str
    score: float
    source: str
    popularity: int = 0
    album_name: str = ""


def _get_spotify_client(ctx: ToolContext):
    """Initializes and returns the Spotipy client using credentials from the config."""
    client_id = ctx.config.spotify_client_id
    client_secret = ctx.config.spotify_client_secret

    if not client_id or not client_secret:
        raise ValueError("Spotify Client ID or Client Secret is missing from the configuration. Please add them to your .butler/config.json file.")

    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    scope = "user-read-playback-state user-modify-playback-state user-read-currently-playing"

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri="http://127.0.0.1:8080",
        scope=scope,
        open_browser=True,
    )

    return spotipy.Spotify(auth_manager=auth_manager)


class SpotifyPlayArgs(BaseModel):
    query: str = Field(description="The name of the song or artist to search for and play.")


class SpotifyPauseArgs(BaseModel):
    pass


class SpotifyStateArgs(BaseModel):
    pass


def _combined_aliases(ctx: ToolContext) -> dict[str, str]:
    aliases: dict[str, str] = {}
    aliases.update(ctx.config.transcript_aliases or {})
    aliases.update(ctx.config.spotify_aliases or {})
    return aliases


def _clean_query(raw_query: str, aliases: dict[str, str]) -> str:
    query = normalize_text(raw_query, aliases=aliases)
    query = _RE_COMMAND_PREFIX.sub("", query).strip()
    return query


def _split_query(query: str) -> tuple[str, str | None]:
    for separator in (" by ", " from ", " - ", " of "):
        if separator in query:
            left, right = query.split(separator, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    return query, None


def _track_text(track: dict[str, Any]) -> tuple[str, str]:
    name = track.get("name", "")
    artists = ", ".join(artist.get("name", "") for artist in track.get("artists", []))
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

    if artist_query:
        score = (0.50 * title_sim) + (0.40 * artist_sim) + (0.10 * query_sim)
    else:
        score = (0.60 * title_sim) + (0.20 * artist_sim) + (0.20 * query_sim)

    lowered = combined.casefold()
    if any(term in lowered for term in (" live", " remix", " karaoke", " cover")) and not any(
        term in raw_query.casefold() for term in ("live", "remix", "karaoke", "cover")
    ):
        score -= 0.08

    if title_query and title_query.casefold() in track_name.casefold():
        score += 0.07
    if artist_query and artist_query.casefold() in artist_name.casefold():
        score += 0.10

    score += min(max(popularity, 0), 100) / 1000.0
    return max(0.0, min(score, 1.0))


def _collect_track_candidates(
    sp: Any,
    *,
    query: str,
    title_query: str,
    artist_query: str | None,
    limit: int,
) -> list[SpotifyCandidate]:
    search_queries = []
    if title_query and artist_query:
        search_queries.extend(
            [
                f'track:"{title_query}" artist:"{artist_query}"',
                f"track:{title_query} artist:{artist_query}",
                f"{title_query} {artist_query}",
            ]
        )
    search_queries.extend([query, title_query or query])
    if artist_query:
        search_queries.append(artist_query)

    candidates: list[SpotifyCandidate] = []
    seen: set[str] = set()
    for search_query in dict.fromkeys(search_queries):
        try:
            result = sp.search(q=search_query, type="track", limit=limit)
        except Exception as e:  # noqa: BLE001
            logger.debug("spotify track search failed query=%s error=%s", search_query, e)
            continue

        tracks = result.get("tracks", {}).get("items", [])
        for track in tracks:
            track_uri = track.get("uri")
            track_name, artist_name = _track_text(track)
            if not track_uri or not track_name:
                continue
            key = f"{track_uri}:{search_query}"
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                SpotifyCandidate(
                    track_uri=track_uri,
                    track_name=track_name,
                    artist_name=artist_name,
                    score=0.0,
                    source=f"track:{search_query}",
                    popularity=int(track.get("popularity") or 0),
                    album_name=track.get("album", {}).get("name", ""),
                )
            )

    return candidates


def _collect_artist_candidates(
    sp: Any,
    *,
    query: str,
    artist_query: str,
    limit: int,
) -> list[SpotifyCandidate]:
    candidates: list[SpotifyCandidate] = []
    try:
        result = sp.search(q=artist_query or query, type="artist", limit=max(3, min(limit, 5)))
    except Exception as e:  # noqa: BLE001
        logger.debug("spotify artist search failed query=%s error=%s", artist_query or query, e)
        return candidates

    artists = result.get("artists", {}).get("items", [])
    for artist in artists[:3]:
        artist_id = artist.get("id")
        artist_name = artist.get("name", "")
        if not artist_id or not artist_name:
            continue

        try:
            top_tracks = sp.artist_top_tracks(artist_id).get("tracks", [])
        except Exception as e:  # noqa: BLE001
            logger.debug("spotify top tracks failed artist=%s error=%s", artist_name, e)
            continue

        for track in top_tracks[:3]:
            track_uri = track.get("uri")
            track_name, artists = _track_text(track)
            if not track_uri or not track_name:
                continue
            candidates.append(
                SpotifyCandidate(
                    track_uri=track_uri,
                    track_name=track_name,
                    artist_name=artists or artist_name,
                    score=0.0,
                    source=f"artist:{artist_name}",
                    popularity=int(track.get("popularity") or 0),
                    album_name=track.get("album", {}).get("name", ""),
                )
            )

    return candidates


def _rank_candidates(
    candidates: list[SpotifyCandidate],
    *,
    raw_query: str,
    title_query: str,
    artist_query: str | None,
) -> list[SpotifyCandidate]:
    ranked: list[SpotifyCandidate] = []
    for candidate in candidates:
        score = _score_track(
            track_name=candidate.track_name,
            artist_name=candidate.artist_name,
            popularity=candidate.popularity,
            raw_query=raw_query,
            title_query=title_query,
            artist_query=artist_query,
            source=candidate.source,
        )
        ranked.append(
            SpotifyCandidate(
                track_uri=candidate.track_uri,
                track_name=candidate.track_name,
                artist_name=candidate.artist_name,
                score=score,
                source=candidate.source,
                popularity=candidate.popularity,
                album_name=candidate.album_name,
            )
        )

    ranked.sort(key=lambda item: (item.score, item.popularity), reverse=True)
    return ranked


def _format_question(candidate: SpotifyCandidate, *, confidence: float) -> dict[str, Any]:
    return {
        "success": True,
        "needs_clarification": True,
        "confidence": confidence,
        "question": f"Did you mean '{candidate.track_name}' by {candidate.artist_name}?",
        "suggestion": {
            "track_name": candidate.track_name,
            "artist": candidate.artist_name,
            "album": candidate.album_name,
        },
    }


def _select_best_candidate(sp: Any, ctx: ToolContext, raw_query: str) -> dict[str, Any]:
    aliases = _combined_aliases(ctx)
    cleaned_query = _clean_query(raw_query, aliases)
    title_query, artist_query = _split_query(cleaned_query)
    search_limit = max(1, min(int(ctx.config.spotify_search_limit), 25))

    candidates = _collect_track_candidates(
        sp,
        query=cleaned_query,
        title_query=title_query,
        artist_query=artist_query,
        limit=search_limit,
    )
    if artist_query:
        candidates.extend(
            _collect_artist_candidates(
                sp,
                query=cleaned_query,
                artist_query=artist_query,
                limit=search_limit,
            )
        )
    else:
        candidates.extend(
            _collect_artist_candidates(
                sp,
                query=cleaned_query,
                artist_query=cleaned_query,
                limit=search_limit,
            )
        )

    ranked = _rank_candidates(
        candidates,
        raw_query=cleaned_query,
        title_query=title_query,
        artist_query=artist_query,
    )

    if not ranked:
        return {"success": False, "error": f"No tracks found matching: '{raw_query}'"}

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    confidence = best.score
    gap = confidence - (runner_up.score if runner_up else 0.0)

    if confidence < ctx.config.spotify_clarify_confidence or (runner_up and confidence < ctx.config.spotify_accept_confidence and gap < 0.12):
        return _format_question(best, confidence=confidence)

    return {
        "success": True,
        "status": f"Now playing: '{best.track_name}' by {best.artist_name}",
        "track_name": best.track_name,
        "artist": best.artist_name,
        "album": best.album_name,
        "confidence": confidence,
        "matched_from": best.source,
        "selected_track_uri": best.track_uri,
    }


def _start_playback(sp: Any, track_uri: str) -> None:
    devices = sp.devices()
    active_devices = [device for device in devices.get("devices", []) if device.get("is_active")]

    if not active_devices and devices.get("devices"):
        device_id = devices["devices"][0]["id"]
        sp.start_playback(device_id=device_id, uris=[track_uri])
    elif not devices.get("devices"):
        raise ValueError("No active Spotify devices found running on your account. Please open the Spotify app on your desktop or phone first.")
    else:
        sp.start_playback(uris=[track_uri])


def handle_spotify_play(ctx: ToolContext, args: SpotifyPlayArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        selection = _select_best_candidate(sp, ctx, args.query)
        if not selection.get("success"):
            return selection
        if selection.get("needs_clarification"):
            return selection

        track_uri = selection["selected_track_uri"]
        _start_playback(sp, track_uri)
        selection.pop("selected_track_uri", None)
        return selection
    except Exception as e:
        logger.error("Spotify play failed: %s", e)
        return {"success": False, "error": str(e)}


def handle_spotify_pause(ctx: ToolContext, args: SpotifyPauseArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        sp.pause_playback()
        return {"success": True, "status": "Spotify playback paused"}
    except Exception as e:
        if "Player command failed: Restriction violated" in str(e):
            return {"success": True, "status": "Spotify is already paused."}
        return {"success": False, "error": str(e)}


def handle_spotify_state(ctx: ToolContext, args: SpotifyStateArgs) -> dict[str, Any]:
    try:
        sp = _get_spotify_client(ctx)
        current = sp.current_playback()

        if current is None or not current.get("is_playing"):
            return {"success": True, "is_playing": False, "status": "Nothing is currently playing on Spotify."}

        item = current.get("item", {})
        if not item:
            return {"success": True, "is_playing": False, "status": "No track information available."}

        track_name = item.get("name", "Unknown Track")
        artist_name = item.get("artists", [{"name": "Unknown Artist"}])[0].get("name")
        album_name = item.get("album", {}).get("name", "Unknown Album")

        return {
            "success": True,
            "is_playing": True,
            "track_name": track_name,
            "artist": artist_name,
            "album": album_name,
            "status": f"Currently playing '{track_name}' by {artist_name} from the album '{album_name}'.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def build() -> list[Tool]:
    return [
        Tool[SpotifyPlayArgs](
            name="spotify.play",
            description="Search for a song or artist on Spotify and instantly start playing it. Requires Spotify Premium.",
            input_model=SpotifyPlayArgs,
            handler=handle_spotify_play,
            side_effect=True,
        ),
        Tool[SpotifyPauseArgs](
            name="spotify.pause",
            description="Pause the current Spotify playback.",
            input_model=SpotifyPauseArgs,
            handler=handle_spotify_pause,
            side_effect=True,
        ),
        Tool[SpotifyStateArgs](
            name="spotify.state",
            description="Check what song is currently playing on Spotify.",
            input_model=SpotifyStateArgs,
            handler=handle_spotify_state,
            side_effect=False,
        ),
    ]
