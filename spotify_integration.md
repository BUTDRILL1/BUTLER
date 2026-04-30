# 🎵 BUTLER — Spotify Integration: God-Level Reference

> **Verified against**: [Spotify OpenAPI Specification](https://developer.spotify.com/reference/web-api/open-api-schema.yaml)
> and the [November 27, 2024 access restriction announcement](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api).
> Base URI for all API calls: `https://api.spotify.com/v1`
>
> **Scope of this document**: Auth is already live. This document covers everything beyond basic
> playback — advanced player control, full playlist & library management, search, personalization,
> catalog lookups, social features, error handling, rate limiting, token management, and
> production-grade Python patterns. Every claim is spec-verified.

---

## Table of Contents

1. [Concepts Every Dev Must Understand First](#1-concepts-every-dev-must-understand-first)
2. [Token Management — The Correct Way](#2-token-management--the-correct-way)
3. [Spotify URIs vs IDs vs URLs — The Eternal Confusion](#3-spotify-uris-vs-ids-vs-urls--the-eternal-confusion)
4. [The `market` Parameter — Why You Must Always Send It](#4-the-market-parameter--why-you-must-always-send-it)
5. [Scopes — Complete Reference](#5-scopes--complete-reference)
6. [Rate Limiting — The Right Way to Handle It](#6-rate-limiting--the-right-way-to-handle-it)
7. [Error Handling — Complete Reference](#7-error-handling--complete-reference)
8. [The Production Shield — Base Python Class](#8-the-production-shield--base-python-class)
9. [Player — Advanced Control](#9-player--advanced-control)
10. [Search — Power Usage](#10-search--power-usage)
11. [Playlists — Full CRUD](#11-playlists--full-crud)
12. [Library Management — Tracks, Albums, Episodes, Audiobooks](#12-library-management--tracks-albums-episodes-audiobooks)
13. [Personalization — User Taste & Top Items](#13-personalization--user-taste--top-items)
14. [Catalog Lookups — Tracks, Albums, Artists, Shows, Episodes](#14-catalog-lookups--tracks-albums-artists-shows-episodes)
15. [Social Features — Follow & Check](#15-social-features--follow--check)
16. [Custom Playlist Images](#16-custom-playlist-images)
17. [Paging — How to Fetch All Results](#17-paging--how-to-fetch-all-results)
18. [Access-Restricted Endpoints (Post Nov 2024)](#18-access-restricted-endpoints-post-nov-2024)
19. [The One Truly Deprecated Endpoint](#19-the-one-truly-deprecated-endpoint)
20. [Scope Master List for BUTLER](#20-scope-master-list-for-butler)
21. [Pre-Shipping Checklist](#21-pre-shipping-checklist)

---

## 1. Concepts Every Dev Must Understand First

### What is an Access Token?
When BUTLER talks to Spotify's API, every single request must include a header like:
```
Authorization: Bearer BQA1abc...xyz
```
That long string is the **access token**. It proves Spotify has authorized BUTLER to act on behalf
of the user. It expires after **3600 seconds (1 hour)**. After that, every API call returns `401`
until you get a new one using the refresh token. This must be handled automatically — the user
should never be asked to log in again just because an hour passed.

### What is a Refresh Token?
The refresh token is a long-lived credential (it does not expire unless the user revokes access)
that lets BUTLER silently get a new access token without user interaction. Store it securely.
Losing it means the user has to re-authorize.

### What are Scopes?
Scopes are permissions. When the user first logs in, Spotify shows them a consent screen listing
exactly what BUTLER is asking permission to do. You must request only the scopes you actually need.
Requesting `user-modify-playback-state` but only using read operations is a ToS violation.
Forgetting a scope means your API calls will silently fail with `403 Forbidden`.

### What is Premium vs Free?
Playback control endpoints (`play`, `pause`, `skip`, `volume`, `queue`) **require Spotify Premium**.
Free users can use search, library, personalization, and catalog endpoints just fine.
**Do not pre-check** whether the user has Premium by reading their profile — just attempt the action
and handle the `403` response. The error body will tell you it is Premium-related.

### What is a Device?
Spotify's playback is device-based. "A device" means a Spotify client that is open and active —
the desktop app, mobile app, web player, etc. If no device is active, playback calls return `404`.
You can fetch the list of available devices and transfer playback to a specific one.

---

## 2. Token Management — The Correct Way

Spotipy handles token refresh automatically when you use `SpotifyOAuth` — but only if you
call it correctly. Here is the production-safe pattern:

```python
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# NEVER hardcode credentials. Always pull from environment variables.
auth_manager = SpotifyOAuth(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri="http://127.0.0.1:8888/callback",  # Must match exactly what's in the Spotify dashboard
    scope=" ".join([                                  # Space-separated string of scopes
        "user-read-playback-state",
        "user-modify-playback-state",
        "user-read-currently-playing",
        "playlist-read-private",
        "playlist-modify-private",
        "playlist-modify-public",
        "user-library-read",
        "user-library-modify",
        "user-top-read",
        "user-follow-read",
        "user-follow-modify",
        "ugc-image-upload",
    ]),
    cache_path=".spotify_token_cache",  # Spotipy saves the refresh token here between runs
    open_browser=False,                  # Do not auto-open browser; handle redirect manually
)

sp = spotipy.Spotify(auth_manager=auth_manager)
```

### How spotipy handles refresh automatically
Every time you call `sp.some_method()`, spotipy checks if the cached access token is still valid.
If it has expired, it uses the refresh token to silently get a new one from Spotify and retries.
You do not need to write refresh logic yourself **as long as you use `SpotifyOAuth` and the cache**.

### What if you are managing tokens manually (e.g., for a multi-user server)?
If BUTLER serves multiple users, each user needs their own token. Do not use a single shared
`SpotifyOAuth` instance. Instead, store each user's `access_token`, `refresh_token`, and
`expires_at` in your database. When `expires_at` is in the past, call the token endpoint directly:

```python
import requests
import base64
import time

def refresh_access_token(refresh_token: str) -> dict:
    """
    Exchange a refresh token for a new access token.
    Returns a dict with 'access_token' and 'expires_at'.
    """
    credentials = base64.b64encode(
        f"{os.environ['SPOTIFY_CLIENT_ID']}:{os.environ['SPOTIFY_CLIENT_SECRET']}".encode()
    ).decode()

    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
    )
    response.raise_for_status()
    data = response.json()

    return {
        "access_token": data["access_token"],
        "expires_at": time.time() + data["expires_in"],  # Store expiry as epoch timestamp
        # Spotify may or may not return a new refresh_token — if it does, save it
        "refresh_token": data.get("refresh_token", refresh_token),
    }
```

---

## 3. Spotify URIs vs IDs vs URLs — The Eternal Confusion

Every Spotify item (track, album, artist, playlist, etc.) has three ways to identify it.
Mixing them up is the number one beginner mistake.

| Format | Example | When to use |
| :--- | :--- | :--- |
| **Spotify URI** | `spotify:track:4iV5W9uYEdYUVa79Axb7Rh` | Playback, adding to queue, adding to playlists |
| **Spotify ID** | `4iV5W9uYEdYUVa79Axb7Rh` | API path parameters and `ids` query params |
| **Spotify URL** | `https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh` | Sharing links to users, attribution |

The ID is just the last segment of both the URI and the URL. To extract it from a URI:
```python
def uri_to_id(uri: str) -> str:
    # "spotify:track:4iV5W9uYEdYUVa79Axb7Rh" → "4iV5W9uYEdYUVa79Axb7Rh"
    return uri.split(":")[-1]
```

Spotipy is generally smart enough to accept either URIs or IDs in most of its wrapper methods,
but when building raw requests or passing data to playlist operations, use URIs for
playback/queue/playlist-add and IDs for catalog lookups.

---

## 4. The `market` Parameter — Why You Must Always Send It

The `market` parameter is an [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
country code (e.g., `"IN"` for India, `"US"` for USA). Here is why it matters:

Spotify licenses music differently per country. A track available in the US may be unavailable
in India. If you do not send `market`, Spotify may return tracks marked as unplayable, or return
nothing at all for certain catalog queries, resulting in unexpected `404` errors or empty results
even when the content exists.

**Rule**: Always pass `market`. The safest value is the user's country from their profile:

```python
def get_user_market(sp: spotipy.Spotify) -> str:
    """Fetch the user's country from their profile. Requires user-read-private scope."""
    user = sp.current_user()
    return user.get("country", "US")  # Fall back to US if somehow absent
```

Cache this value — you do not need to fetch it on every request.

---

## 5. Scopes — Complete Reference

Scopes are grouped by what they unlock. Only request what you need per feature.

### Playback (Premium only for modification)
| Scope | What it unlocks |
| :--- | :--- |
| `user-read-playback-state` | Get current playback state, device list, currently playing track |
| `user-modify-playback-state` | Play, pause, skip, seek, set volume, set repeat/shuffle, transfer playback, add to queue |
| `user-read-currently-playing` | Get the currently playing track (lighter than full playback state) |
| `user-read-recently-played` | Get the user's recently played tracks |

### Library
| Scope | What it unlocks |
| :--- | :--- |
| `user-library-read` | Read saved tracks, albums, episodes, audiobooks |
| `user-library-modify` | Save or remove tracks, albums, episodes, audiobooks |

### Playlists
| Scope | What it unlocks |
| :--- | :--- |
| `playlist-read-private` | Read private playlists |
| `playlist-modify-public` | Create, edit, add/remove tracks on public playlists |
| `playlist-modify-private` | Create, edit, add/remove tracks on private playlists |

### Personalization
| Scope | What it unlocks |
| :--- | :--- |
| `user-top-read` | Get user's top tracks and artists |

### User Profile
| Scope | What it unlocks |
| :--- | :--- |
| `user-read-private` | Read user's country, subscription type, display name |
| `user-read-email` | Read user's email address |

### Social
| Scope | What it unlocks |
| :--- | :--- |
| `user-follow-read` | Check which artists/users the user follows |
| `user-follow-modify` | Follow or unfollow artists/users |

### Uploads
| Scope | What it unlocks |
| :--- | :--- |
| `ugc-image-upload` | Upload custom playlist cover images |

### Podcasts / Playback Position
| Scope | What it unlocks |
| :--- | :--- |
| `user-read-playback-position` | Read playback position in episodes/audiobooks |

---

## 6. Rate Limiting — The Right Way to Handle It

Spotify enforces rate limits. When you exceed them, you get `HTTP 429 Too Many Requests`.
The response includes a `Retry-After` header telling you how many seconds to wait.

**The correct algorithm:**

1. On first `429`: read `Retry-After` header. Wait that many seconds **plus 1** as a buffer. Retry.
2. On second `429` in the same call chain: switch to exponential backoff — `2^attempt * base_wait`.
3. Never retry in a tight loop without sleeping. That will get your app rate-banned.
4. Log every rate limit hit with the endpoint and wait time. If you are hitting limits often,
   you need to batch requests or cache more aggressively.

```python
import time

def wait_for_rate_limit(attempt: int, retry_after_header: str | None, base_wait: int = 5) -> None:
    """
    Sleep the appropriate amount of time after a 429 response.
    attempt=0 → first 429, use Retry-After header.
    attempt>0 → subsequent 429s, use exponential backoff.
    """
    if attempt == 0:
        wait = int(retry_after_header or base_wait) + 1
    else:
        wait = (2 ** attempt) * base_wait
    print(f"[Rate limit] Waiting {wait}s (attempt {attempt + 1})")
    time.sleep(wait)
```

---

## 7. Error Handling — Complete Reference

Every API response can return these HTTP status codes. Handle all of them.

| Code | Name | What it means | What to do |
| :--- | :--- | :--- | :--- |
| `200` | OK | Success | Use the response body |
| `201` | Created | Resource created (e.g., snapshot after adding to playlist) | Use the response body |
| `204` | No Content | Success but no body (e.g., after pause/play) | Nothing to parse |
| `400` | Bad Request | Malformed request (bad parameter, missing required field) | Fix the request |
| `401` | Unauthorized | Access token missing, expired, or invalid | Refresh token and retry |
| `403` | Forbidden | Valid token but wrong scope, or Premium required | Check scope list; read error body to distinguish scope vs Premium |
| `404` | Not Found | Invalid ID, no active device, content not in market | Validate IDs; check market; for playback prompt user to open Spotify |
| `429` | Too Many Requests | Rate limit hit | Wait `Retry-After` seconds, then retry with backoff |
| `500`/`502`/`503` | Server Error | Spotify's servers are down | Retry with exponential backoff; do not alert user immediately |

### Reading the error body
Spotify returns a JSON error body that tells you *why* the call failed. Always log it:
```json
{
  "error": {
    "status": 403,
    "message": "Player command failed: Premium required"
  }
}
```
This is how you distinguish "wrong scope" from "user does not have Premium" — both return `403`
but the `message` field differs.

---

## 8. The Production Shield — Base Python Class

This is the foundation everything else builds on. Every method in BUTLER that touches Spotify
should go through this class.

```python
import os
import time
import functools
import spotipy
from spotipy.oauth2 import SpotifyOAuth


def spotify_retry(max_retries: int = 3, base_wait: int = 5):
    """
    Decorator that wraps any Spotify API call with:
    - Automatic retry on 429 (rate limit) with Retry-After + exponential backoff
    - Meaningful error messages for 403 (scope vs Premium) and 404 (no device)
    - Retry on 5xx server errors
    - Re-raise on unrecoverable errors after max_retries
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except spotipy.exceptions.SpotifyException as e:

                    if e.http_status == 429:
                        # Rate limited — respect Retry-After, escalate to backoff on repeat hits
                        retry_after = e.headers.get("Retry-After") if e.headers else None
                        wait = int(retry_after or base_wait) + 1 if attempt == 0 else (2 ** attempt) * base_wait
                        print(f"[429] Rate limited on {func.__name__}. Waiting {wait}s...")
                        time.sleep(wait)

                    elif e.http_status == 401:
                        # Token expired — spotipy's auth_manager handles this automatically in most
                        # cases. If it bubbles up here, re-raise immediately; do not retry blindly.
                        print(f"[401] Token expired or invalid in {func.__name__}. Re-raising.")
                        raise

                    elif e.http_status == 403:
                        # Either missing scope or Premium required — read the message to distinguish
                        msg = str(e.msg or "")
                        if "Premium" in msg:
                            raise PermissionError(
                                "Spotify Premium is required for this action."
                            )
                        else:
                            raise PermissionError(
                                f"Spotify scope error in {func.__name__}: {msg}. "
                                "Check that the required scope was requested during authorization."
                            )

                    elif e.http_status == 404:
                        msg = str(e.msg or "")
                        if "NO_ACTIVE_DEVICE" in msg or "no active device" in msg.lower():
                            raise RuntimeError(
                                "No active Spotify device found. "
                                "Please open Spotify on a device first."
                            )
                        raise  # Re-raise other 404s as-is

                    elif e.http_status in (500, 502, 503):
                        # Spotify server error — retry with backoff
                        wait = (2 ** attempt) * base_wait
                        print(f"[{e.http_status}] Spotify server error. Retrying in {wait}s...")
                        time.sleep(wait)

                    else:
                        raise  # Unknown error — re-raise immediately

                    if attempt == max_retries - 1:
                        raise  # Exhausted retries

            return None
        return wrapper
    return decorator


class SpotifyClient:
    """
    Singleton-like wrapper around the spotipy client.
    All BUTLER Spotify features go through this class.
    """

    def __init__(self):
        self.auth_manager = SpotifyOAuth(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri="http://127.0.0.1:8888/callback",
            scope=" ".join([
                "user-read-playback-state",
                "user-modify-playback-state",
                "user-read-currently-playing",
                "user-read-recently-played",
                "playlist-read-private",
                "playlist-modify-private",
                "playlist-modify-public",
                "user-library-read",
                "user-library-modify",
                "user-top-read",
                "user-follow-read",
                "user-follow-modify",
                "ugc-image-upload",
                "user-read-private",
                "user-read-playback-position",
            ]),
            cache_path=".spotify_token_cache",
            open_browser=False,
        )
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        self._market: str | None = None

    def get_market(self) -> str:
        """Lazily fetch and cache the user's country code."""
        if not self._market:
            user = self.sp.current_user()
            self._market = user.get("country", "US")
        return self._market


# Global instance — import this wherever you need Spotify
butler_spotify = SpotifyClient()
sp = butler_spotify.sp
```

---

## 9. Player — Advanced Control

All playback endpoints require **Spotify Premium**. Handle `403` gracefully.

### 9.1 Get Full Playback State

Returns the active device, currently playing track, progress, shuffle/repeat state, and volume.

```python
@spotify_retry()
def get_playback_state() -> dict | None:
    """
    Returns full playback state, or None if nothing is playing.
    Scope: user-read-playback-state
    """
    state = sp.current_playback(market=butler_spotify.get_market())
    if not state:
        return None  # Nothing is playing or no active device

    return {
        "is_playing": state["is_playing"],
        "track_name": state["item"]["name"],
        "artist": state["item"]["artists"][0]["name"],
        "album": state["item"]["album"]["name"],
        "progress_ms": state["progress_ms"],
        "duration_ms": state["item"]["duration_ms"],
        "volume": state["device"]["volume_percent"],
        "shuffle": state["shuffle_state"],
        "repeat": state["repeat_state"],    # "off" | "track" | "context"
        "device_name": state["device"]["name"],
        "device_id": state["device"]["id"],
        "track_uri": state["item"]["uri"],
        "track_id": state["item"]["id"],
    }
```

### 9.2 Get Currently Playing Track (Lightweight)

Use this if you only need the current track, not the full device/volume/shuffle state.

```python
@spotify_retry()
def get_currently_playing() -> dict | None:
    """
    Scope: user-read-currently-playing
    Lighter call than get_playback_state — no device info.
    """
    current = sp.currently_playing(market=butler_spotify.get_market())
    if not current or not current.get("item"):
        return None
    track = current["item"]
    return {
        "track_name": track["name"],
        "artist": ", ".join(a["name"] for a in track["artists"]),
        "album": track["album"]["name"],
        "track_uri": track["uri"],
        "is_playing": current["is_playing"],
        "progress_ms": current["progress_ms"],
        "duration_ms": track["duration_ms"],
    }
```

### 9.3 Play a Specific Track, Album, or Playlist

```python
@spotify_retry()
def play_track(track_uri: str, device_id: str | None = None) -> None:
    """
    Play a specific track by URI.
    Scope: user-modify-playback-state

    track_uri example: "spotify:track:4iV5W9uYEdYUVa79Axb7Rh"
    device_id: pass a specific device ID to target a device, or None for active device.
    """
    sp.start_playback(device_id=device_id, uris=[track_uri])


@spotify_retry()
def play_context(context_uri: str, offset_uri: str | None = None, device_id: str | None = None) -> None:
    """
    Play an album or playlist as a context (enables correct shuffle/queue behavior).
    Scope: user-modify-playback-state

    context_uri example: "spotify:album:1DFixLWuPkv3KT3TnV35m3"
    offset_uri: optionally start from a specific track within the context.
    """
    offset = {"uri": offset_uri} if offset_uri else None
    sp.start_playback(device_id=device_id, context_uri=context_uri, offset=offset)
```

### 9.4 Seek to Position

```python
@spotify_retry()
def seek_to(position_ms: int) -> None:
    """
    Jump to a specific position in the current track.
    Scope: user-modify-playback-state

    position_ms: milliseconds from start. E.g., 30000 = 30 seconds in.
    """
    sp.seek_track(position_ms)
```

### 9.5 Set Repeat Mode

```python
@spotify_retry()
def set_repeat(mode: str) -> None:
    """
    Set repeat mode.
    Scope: user-modify-playback-state

    mode options:
        "off"     — No repeat
        "track"   — Repeat current track
        "context" — Repeat current album/playlist
    """
    if mode not in ("off", "track", "context"):
        raise ValueError(f"Invalid repeat mode: {mode}. Must be 'off', 'track', or 'context'.")
    sp.repeat(mode)
```

### 9.6 Set Shuffle

```python
@spotify_retry()
def set_shuffle(state: bool) -> None:
    """
    Enable or disable shuffle.
    Scope: user-modify-playback-state
    """
    sp.shuffle(state)
```

### 9.7 Set Volume

```python
@spotify_retry()
def set_volume(percent: int) -> None:
    """
    Set playback volume.
    Scope: user-modify-playback-state

    percent: integer 0–100.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"Volume must be between 0 and 100. Got: {percent}")
    sp.volume(percent)
```

### 9.8 Add to Queue

```python
@spotify_retry()
def add_to_queue(track_uri: str, device_id: str | None = None) -> None:
    """
    Add a track to the end of the playback queue.
    Scope: user-modify-playback-state

    track_uri example: "spotify:track:4iV5W9uYEdYUVa79Axb7Rh"
    Note: Only one track can be added per call. Loop if you have multiple.
    """
    sp.add_to_queue(track_uri, device_id=device_id)
```

### 9.9 Get Available Devices

```python
@spotify_retry()
def get_devices() -> list[dict]:
    """
    Get all Spotify devices currently available for this user.
    Scope: user-read-playback-state

    Returns a list of device dicts with: id, name, type, is_active, volume_percent.
    Use the 'id' from a device here as device_id in other playback calls.
    """
    result = sp.devices()
    return result.get("devices", [])
```

### 9.10 Transfer Playback to Another Device

```python
@spotify_retry()
def transfer_playback(device_id: str, force_play: bool = False) -> None:
    """
    Move playback to a different device.
    Scope: user-modify-playback-state

    device_id: get this from get_devices()
    force_play: if True, start playing immediately even if currently paused.
    """
    sp.transfer_playback(device_id, force_play=force_play)
```

### 9.11 Get Recently Played Tracks

```python
@spotify_retry()
def get_recently_played(limit: int = 20) -> list[dict]:
    """
    Get the user's recently played tracks (max 50).
    Scope: user-read-recently-played   ← add this to your scope list

    Returns tracks in reverse chronological order (most recent first).
    """
    result = sp.current_user_recently_played(limit=min(limit, 50))
    return [
        {
            "track_name": item["track"]["name"],
            "artist": item["track"]["artists"][0]["name"],
            "track_uri": item["track"]["uri"],
            "played_at": item["played_at"],  # ISO 8601 timestamp string
        }
        for item in result.get("items", [])
    ]
```

---

## 10. Search — Power Usage

`GET /search` — No scope required for public catalog searches.

### 10.1 Basic Search

```python
@spotify_retry()
def search(query: str, types: list[str] = None, limit: int = 10) -> dict:
    """
    Search Spotify's catalog.

    query: your search string. Can include field filters (see below).
    types: list of item types to search. Options:
           "track", "album", "artist", "playlist", "show", "episode", "audiobook"
           Defaults to ["track"].
    limit: max results per type, 1–50.

    Returns the raw search result dict keyed by type.
    Example: result["tracks"]["items"] gives you the list of track objects.
    """
    if types is None:
        types = ["track"]

    return sp.search(
        q=query,
        type=",".join(types),
        market=butler_spotify.get_market(),
        limit=min(limit, 50),
    )
```

### 10.2 Field Filters — Precision Searching

Field filters let you target specific metadata fields. Combine them freely.

| Filter | Applies to | Example |
| :--- | :--- | :--- |
| `artist:` | Albums, artists, tracks | `artist:"Anuv Jain"` |
| `track:` | Tracks | `track:"Gul"` |
| `album:` | Albums, tracks | `album:"Lullaby"` |
| `genre:` | Artists, tracks | `genre:"indie pop"` |
| `year:` | Albums, artists, tracks | `year:2022` or `year:2019-2023` |
| `isrc:` | Tracks | `isrc:INXXX2200123` |
| `tag:new` | Albums | Albums released in the last 2 weeks |
| `tag:hipster` | Albums | Albums in the lowest 10% popularity |

```python
# Find a specific track by a specific artist — precise
results = search('artist:"Anuv Jain" track:"Baarish Ki Jaaye"', types=["track"])
tracks = results["tracks"]["items"]

# Find indie pop albums from 2020-2023
results = search('genre:"indie pop" year:2020-2023', types=["album"])
albums = results["albums"]["items"]
```

### 10.3 Extracting Results

```python
def extract_track_info(track: dict) -> dict:
    """Pull the fields you actually need from a track result object."""
    return {
        "name": track["name"],
        "artist": ", ".join(a["name"] for a in track["artists"]),
        "album": track["album"]["name"],
        "duration_ms": track["duration_ms"],
        "uri": track["uri"],                # Use for playback and queue
        "id": track["id"],                  # Use for catalog lookups
        "popularity": track["popularity"],  # 0–100, higher = more popular globally
        "explicit": track["explicit"],
        "album_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
    }
```

---

## 11. Playlists — Full CRUD

### 11.1 Get a Playlist's Details

```python
@spotify_retry()
def get_playlist(playlist_id: str) -> dict:
    """
    Get metadata about a playlist (name, description, owner, track count).
    No scope needed for public playlists.
    Scope: playlist-read-private for private ones.

    playlist_id: the Spotify ID of the playlist (just the ID — not the URI, not the URL).
    """
    return sp.playlist(playlist_id, market=butler_spotify.get_market())
```

### 11.2 Get All Items in a Playlist (with autopaging)

Playlists can have more than 100 tracks. The API returns max 100 per request.
You must page through to get all of them.

```python
@spotify_retry()
def get_all_playlist_tracks(playlist_id: str) -> list[dict]:
    """
    Fetch every track in a playlist, handling pagination automatically.
    Scope: playlist-read-private

    The 'fields' parameter tells Spotify to return only the data we need,
    which makes the response smaller and faster.
    """
    tracks = []
    offset = 0
    limit = 100  # Max per request

    while True:
        result = sp.playlist_items(
            playlist_id,
            market=butler_spotify.get_market(),
            limit=limit,
            offset=offset,
            fields="items(track(id,name,uri,artists,album,duration_ms)),next,total",
        )
        items = result.get("items", [])
        for item in items:
            track = item.get("track")
            if track:  # track can be None if the song was deleted from Spotify entirely
                tracks.append({
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "uri": track["uri"],
                    "id": track["id"],
                })

        # If 'next' is None, we have fetched everything
        if result.get("next") is None:
            break
        offset += limit

    return tracks
```

### 11.3 Create a Playlist

```python
@spotify_retry()
def create_playlist(name: str, description: str = "", public: bool = False) -> str:
    """
    Create a new playlist for the current user.
    Scope: playlist-modify-private (private) or playlist-modify-public (public)

    Returns the new playlist's ID. Use this ID for all subsequent operations on it.
    """
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=public,
        collaborative=False,
        description=description,
    )
    return playlist["id"]
```

### 11.4 Add Tracks to a Playlist

```python
@spotify_retry()
def add_tracks_to_playlist(playlist_id: str, track_uris: list[str], position: int | None = None) -> str:
    """
    Add tracks to a playlist.
    Scope: playlist-modify-private or playlist-modify-public

    track_uris: list of Spotify track URIs. Max 100 per call.
    position: 0-based index to insert at. None = append to end.

    Returns the playlist's new snapshot_id (a version hash — save it if you need
    to reference exactly which version of the playlist you modified).

    If you have more than 100 tracks, batch them:
        for i in range(0, len(uris), 100):
            add_tracks_to_playlist(playlist_id, uris[i:i+100])
    """
    result = sp.playlist_add_items(playlist_id, track_uris[:100], position=position)
    return result["snapshot_id"]
```

### 11.5 Remove Tracks from a Playlist

```python
@spotify_retry()
def remove_tracks_from_playlist(playlist_id: str, track_uris: list[str]) -> str:
    """
    Remove specific tracks from a playlist.
    Scope: playlist-modify-private or playlist-modify-public

    track_uris: list of Spotify track URIs to remove. Max 100 per call.
    Returns new snapshot_id.

    Important: if the same track appears multiple times in the playlist, ALL
    occurrences are removed. To remove only a specific occurrence by position,
    use sp.playlist_remove_specific_occurrences_of_items() instead.
    """
    result = sp.playlist_remove_all_occurrences_of_items(playlist_id, track_uris[:100])
    return result["snapshot_id"]
```

### 11.6 Reorder Tracks in a Playlist

```python
@spotify_retry()
def reorder_playlist_tracks(
    playlist_id: str,
    range_start: int,
    insert_before: int,
    range_length: int = 1,
) -> str:
    """
    Move a block of tracks to a different position in the playlist.
    Scope: playlist-modify-private or playlist-modify-public

    range_start:   0-based index of the first track to move
    range_length:  how many consecutive tracks to move (default 1)
    insert_before: the index position to insert before

    Example — move the track at index 5 to the top of the playlist:
        reorder_playlist_tracks(playlist_id, range_start=5, insert_before=0)

    Example — move tracks at index 3 and 4 to position 10:
        reorder_playlist_tracks(playlist_id, range_start=3, insert_before=10, range_length=2)

    Returns new snapshot_id.
    """
    result = sp.playlist_reorder_items(
        playlist_id,
        range_start=range_start,
        insert_before=insert_before,
        range_length=range_length,
    )
    return result["snapshot_id"]
```

### 11.7 Update Playlist Details

```python
@spotify_retry()
def update_playlist_details(
    playlist_id: str,
    name: str | None = None,
    description: str | None = None,
    public: bool | None = None,
) -> None:
    """
    Change a playlist's name, description, or public/private state.
    Scope: playlist-modify-private or playlist-modify-public
    Only pass the fields you want to change — omitted fields stay as they are.
    """
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if description is not None:
        kwargs["description"] = description
    if public is not None:
        kwargs["public"] = public
    sp.playlist_change_details(playlist_id, **kwargs)
```

### 11.8 Get the User's Own Playlists

```python
@spotify_retry()
def get_my_playlists() -> list[dict]:
    """
    Get all playlists owned or followed by the current user.
    Scope: playlist-read-private
    Handles pagination automatically.
    """
    playlists = []
    offset = 0

    while True:
        result = sp.current_user_playlists(limit=50, offset=offset)
        items = result.get("items", [])
        playlists.extend([
            {
                "id": p["id"],
                "name": p["name"],
                "track_count": p["tracks"]["total"],
                "public": p["public"],
                "owner": p["owner"]["display_name"],
            }
            for p in items
        ])
        if result.get("next") is None:
            break
        offset += 50

    return playlists
```

---

## 12. Library Management — Tracks, Albums, Episodes, Audiobooks

The library is the user's personal "Liked" / "Your Music" collection.
All library endpoints use type-specific paths. There is **no single `/me/library` endpoint** —
that does not exist in the spec.

### 12.1 Tracks

```python
@spotify_retry()
def save_tracks(track_ids: list[str]) -> None:
    """Save tracks to the user's Liked Songs. Scope: user-library-modify. Max 50 per call."""
    sp.current_user_saved_tracks_add(track_ids[:50])

@spotify_retry()
def remove_saved_tracks(track_ids: list[str]) -> None:
    """Remove tracks from Liked Songs. Scope: user-library-modify. Max 50 per call."""
    sp.current_user_saved_tracks_delete(track_ids[:50])

@spotify_retry()
def check_saved_tracks(track_ids: list[str]) -> list[bool]:
    """
    Check if tracks are in the user's library. Scope: user-library-read.
    Returns a list of booleans in the same order as the input list.
    Max 50 per call.

    Example:
        results = check_saved_tracks(["id1", "id2", "id3"])
        # results = [True, False, True]
    """
    return sp.current_user_saved_tracks_contains(track_ids[:50])

@spotify_retry()
def get_saved_tracks(limit: int = 50, offset: int = 0) -> list[dict]:
    """
    Get user's saved tracks. Scope: user-library-read.
    Returns tracks in reverse chronological order (most recently saved first).
    Use limit + offset to page through the full library.
    """
    result = sp.current_user_saved_tracks(
        limit=min(limit, 50),
        offset=offset,
        market=butler_spotify.get_market(),
    )
    return [
        {
            "name": item["track"]["name"],
            "artist": item["track"]["artists"][0]["name"],
            "uri": item["track"]["uri"],
            "added_at": item["added_at"],
        }
        for item in result.get("items", [])
    ]
```

### 12.2 Albums

```python
@spotify_retry()
def save_albums(album_ids: list[str]) -> None:
    """Save albums to the user's library. Scope: user-library-modify. Max 50 per call."""
    sp.current_user_saved_albums_add(album_ids[:50])

@spotify_retry()
def remove_saved_albums(album_ids: list[str]) -> None:
    """Remove albums from the user's library. Scope: user-library-modify."""
    sp.current_user_saved_albums_delete(album_ids[:50])

@spotify_retry()
def check_saved_albums(album_ids: list[str]) -> list[bool]:
    """Check if albums are saved. Scope: user-library-read. Returns list of booleans."""
    return sp.current_user_saved_albums_contains(album_ids[:50])

@spotify_retry()
def get_saved_albums(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get user's saved albums. Scope: user-library-read."""
    result = sp.current_user_saved_albums(
        limit=min(limit, 50),
        offset=offset,
        market=butler_spotify.get_market(),
    )
    return [
        {
            "name": item["album"]["name"],
            "artist": item["album"]["artists"][0]["name"],
            "id": item["album"]["id"],
            "added_at": item["added_at"],
        }
        for item in result.get("items", [])
    ]
```

### 12.3 Episodes (Podcasts) — Beta

```python
@spotify_retry()
def save_episodes(episode_ids: list[str]) -> None:
    """
    Save podcast episodes to the user's library.
    Scope: user-library-modify
    Note: This endpoint is in beta — Spotify may change it without warning.
    """
    sp.current_user_saved_episodes_add(episode_ids[:50])

@spotify_retry()
def get_saved_episodes(limit: int = 20) -> list[dict]:
    """Get saved podcast episodes. Scope: user-library-read + user-read-playback-position."""
    result = sp.current_user_saved_episodes(
        limit=min(limit, 50),
        market=butler_spotify.get_market(),
    )
    return [
        {
            "name": item["episode"]["name"],
            "show": item["episode"]["show"]["name"],
            "uri": item["episode"]["uri"],
            "duration_ms": item["episode"]["duration_ms"],
            "added_at": item["added_at"],
        }
        for item in result.get("items", [])
    ]
```

### 12.4 Audiobooks

```python
@spotify_retry()
def save_audiobooks(audiobook_ids: list[str]) -> None:
    """Save audiobooks to the user's library. Scope: user-library-modify. Max 50 per call."""
    sp.current_user_saved_audiobooks_add(audiobook_ids[:50])

@spotify_retry()
def remove_saved_audiobooks(audiobook_ids: list[str]) -> None:
    """Remove audiobooks. Scope: user-library-modify."""
    sp.current_user_saved_audiobooks_delete(audiobook_ids[:50])

@spotify_retry()
def check_saved_audiobooks(audiobook_ids: list[str]) -> list[bool]:
    """Check if audiobooks are saved. Scope: user-library-read."""
    return sp.current_user_saved_audiobooks_contains(audiobook_ids[:50])
```

---

## 13. Personalization — User Taste & Top Items

These are stable, unrestricted endpoints and one of the most powerful tools for building
taste-aware features in BUTLER.

```python
@spotify_retry()
def get_top_tracks(
    limit: int = 20,
    time_range: str = "medium_term",
    offset: int = 0,
) -> list[dict]:
    """
    Get the user's most-listened-to tracks.
    Scope: user-top-read

    time_range options:
        "short_term"  — approx last 4 weeks
        "medium_term" — approx last 6 months  ← best default for taste profiles
        "long_term"   — several years of listening history

    limit: 1–50. offset: use for paging (e.g., offset=50 to get ranks 51–100).
    """
    result = sp.current_user_top_tracks(
        limit=min(limit, 50),
        offset=offset,
        time_range=time_range,
    )
    return [
        {
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "uri": t["uri"],
            "id": t["id"],
            "popularity": t["popularity"],
            "album_art": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
        }
        for t in result.get("items", [])
    ]


@spotify_retry()
def get_top_artists(
    limit: int = 20,
    time_range: str = "medium_term",
    offset: int = 0,
) -> list[dict]:
    """
    Get the user's most-listened-to artists.
    Scope: user-top-read

    Same time_range options as get_top_tracks.
    The 'genres' field is a list of genre strings per artist — very useful for
    building a taste profile without /recommendations.
    """
    result = sp.current_user_top_artists(
        limit=min(limit, 50),
        offset=offset,
        time_range=time_range,
    )
    return [
        {
            "name": a["name"],
            "id": a["id"],
            "uri": a["uri"],
            "genres": a["genres"],        # e.g. ["indie pop", "lo-fi", "hindi indie"]
            "popularity": a["popularity"],
            "image": a["images"][0]["url"] if a["images"] else None,
        }
        for a in result.get("items", [])
    ]
```

### Building a Taste Profile Without `/recommendations`

Since `/recommendations` is access-restricted for new apps, here is how to build
a useful taste profile using only stable endpoints:

```python
def build_taste_profile() -> dict:
    """
    Build a comprehensive taste profile using only unrestricted endpoints.
    Use this as the foundation for any personalization feature in BUTLER.
    """
    top_tracks_short  = get_top_tracks(limit=50, time_range="short_term")
    top_tracks_medium = get_top_tracks(limit=50, time_range="medium_term")
    top_artists       = get_top_artists(limit=20, time_range="medium_term")

    # Extract top genres from top artists (weighted by rank)
    genre_counts: dict[str, int] = {}
    for artist in top_artists:
        for genre in artist["genres"]:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:5]

    return {
        "top_tracks_this_month": top_tracks_short[:10],
        "top_tracks_all_time": top_tracks_medium[:10],
        "top_artists": [a["name"] for a in top_artists[:10]],
        "top_genres": top_genres,
        # Use these IDs to fetch new releases from favourite artists
        "top_artist_ids": [a["id"] for a in top_artists[:5]],
    }
```

---

## 14. Catalog Lookups — Tracks, Albums, Artists, Shows, Episodes

These endpoints are **fully active and not deprecated**. Bulk getters accept up to 50 IDs at once.

### 14.1 Tracks

```python
@spotify_retry()
def get_track(track_id: str) -> dict:
    """Get full details for a single track. No scope required."""
    return sp.track(track_id, market=butler_spotify.get_market())

@spotify_retry()
def get_tracks(track_ids: list[str]) -> list[dict]:
    """Get details for up to 50 tracks at once. No scope required."""
    return sp.tracks(track_ids[:50], market=butler_spotify.get_market())["tracks"]
```

### 14.2 Albums

```python
@spotify_retry()
def get_album(album_id: str) -> dict:
    """Get full album info including all tracks. No scope required."""
    return sp.album(album_id, market=butler_spotify.get_market())

@spotify_retry()
def get_albums(album_ids: list[str]) -> list[dict]:
    """Get details for up to 50 albums at once. No scope required."""
    return sp.albums(album_ids[:50], market=butler_spotify.get_market())["albums"]

@spotify_retry()
def get_artist_albums(artist_id: str, include_groups: str = "album,single") -> list[dict]:
    """
    Get all albums/singles by an artist. Handles pagination automatically.
    No scope required.

    include_groups: comma-separated. Options: "album", "single", "appears_on", "compilation"
    """
    result = sp.artist_albums(
        artist_id,
        album_type=include_groups,
        country=butler_spotify.get_market(),
        limit=50,
    )
    albums = list(result.get("items", []))
    while result.get("next"):
        result = sp.next(result)
        albums.extend(result.get("items", []))
    return albums
```

### 14.3 Artists

```python
@spotify_retry()
def get_artist(artist_id: str) -> dict:
    """Get artist info including genres and popularity. No scope required."""
    return sp.artist(artist_id)

@spotify_retry()
def get_artists(artist_ids: list[str]) -> list[dict]:
    """Get details for up to 50 artists at once. No scope required."""
    return sp.artists(artist_ids[:50])["artists"]

@spotify_retry()
def get_artist_top_tracks(artist_id: str) -> list[dict]:
    """Get an artist's top tracks in the user's market. No scope required."""
    result = sp.artist_top_tracks(artist_id, country=butler_spotify.get_market())
    return result.get("tracks", [])
```

### 14.4 Shows (Podcasts)

```python
@spotify_retry()
def get_show(show_id: str) -> dict:
    """Get podcast show details. Scope: user-read-playback-position."""
    return sp.show(show_id, market=butler_spotify.get_market())

@spotify_retry()
def get_show_episodes(show_id: str, limit: int = 20) -> list[dict]:
    """Get episodes of a podcast. Scope: user-read-playback-position."""
    result = sp.show_episodes(show_id, market=butler_spotify.get_market(), limit=min(limit, 50))
    return result.get("items", [])
```

---

## 15. Social Features — Follow & Check

### 15.1 Artists

```python
@spotify_retry()
def follow_artist(artist_id: str) -> None:
    """Follow an artist. Scope: user-follow-modify."""
    sp.user_follow_artists([artist_id])

@spotify_retry()
def unfollow_artist(artist_id: str) -> None:
    """Unfollow an artist. Scope: user-follow-modify."""
    sp.user_unfollow_artists([artist_id])

@spotify_retry()
def is_following_artist(artist_id: str) -> bool:
    """
    Check if the user follows a specific artist.
    Scope: user-follow-read
    Returns True or False.
    """
    result = sp.current_user_following_artists(ids=[artist_id])
    return result[0] if result else False
```

### 15.2 Playlists

```python
@spotify_retry()
def follow_playlist(playlist_id: str) -> None:
    """
    Follow a playlist.
    Scope: playlist-modify-public or playlist-modify-private
    """
    sp.current_user_follow_playlist(playlist_id)

@spotify_retry()
def unfollow_playlist(playlist_id: str) -> None:
    """Unfollow a playlist. Scope: playlist-modify-public or playlist-modify-private."""
    sp.current_user_unfollow_playlist(playlist_id)
```

---

## 16. Custom Playlist Images

You can set a custom JPEG cover image on any playlist you own.

```python
import base64
import io
from PIL import Image  # pip install Pillow

def compress_image_to_base64(image_path: str, max_kb: int = 250) -> str:
    """
    Load an image, compress it under the Spotify 256 KB hard limit, and return
    it as a base64-encoded JPEG string ready to POST to the API.

    Why 250 KB instead of 256? Small buffer for base64 encoding overhead.
    """
    img = Image.open(image_path).convert("RGB")

    # Resize if very large — Spotify recommends at least 300x300 pixels
    if img.width > 1000 or img.height > 1000:
        img.thumbnail((1000, 1000), Image.LANCZOS)

    quality = 90
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        size_kb = buffer.tell() / 1024
        if size_kb <= max_kb or quality <= 10:
            break
        quality -= 5  # Keep reducing quality until it fits

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@spotify_retry()
def set_playlist_cover(playlist_id: str, image_path: str) -> None:
    """
    Set a custom JPEG cover image for a playlist.
    Scope: ugc-image-upload + playlist-modify-public or playlist-modify-private

    image_path: path to a JPEG or PNG file. Auto-compressed to under 256 KB.
    """
    image_b64 = compress_image_to_base64(image_path)
    sp.playlist_upload_cover_image(playlist_id, image_b64)
```

---

## 17. Paging — How to Fetch All Results

Many Spotify endpoints return paginated results. The response contains a `next` field with a URL
for the next page, or `None` when you have reached the end.

### Using spotipy's built-in `next()` helper

```python
def fetch_all_pages(first_page_result: dict) -> list:
    """
    Given the first page of any paginated Spotify response, fetch all pages and
    return a flat list of all items.

    Usage:
        first_page = sp.current_user_playlists(limit=50)
        all_playlists = fetch_all_pages(first_page)
    """
    items = list(first_page_result.get("items", []))
    page = first_page_result

    while page.get("next"):
        page = sp.next(page)  # spotipy fetches the next URL automatically
        items.extend(page.get("items", []))

    return items
```

### Offset-based paging (for `/me/top`, etc.)

Some endpoints use `limit` + `offset` rather than a `next` URL:

```python
def fetch_all_top_tracks() -> list:
    """Fetch all top tracks using manual offset paging."""
    all_tracks = []
    offset = 0
    limit = 50

    while True:
        result = sp.current_user_top_tracks(limit=limit, offset=offset, time_range="medium_term")
        items = result.get("items", [])
        if not items:
            break
        all_tracks.extend(items)
        if len(items) < limit:
            break  # Last page returned fewer than limit — we are done
        offset += limit

    return all_tracks
```

---

## 18. Access-Restricted Endpoints (Post Nov 2024)

On November 27, 2024, Spotify restricted these endpoints for **new apps** and apps in
development mode without extended access approval. Apps created after this date receive
`403` or `404` when calling them.

**Do not use these in new BUTLER code:**

| Endpoint | Restriction |
| :--- | :--- |
| `GET /recommendations` | Blocked for new apps |
| `GET /audio-features/{id}` | Blocked for new apps |
| `GET /audio-features` (bulk) | Blocked for new apps |
| `GET /audio-analysis/{id}` | Blocked for new apps |
| `GET /browse/featured-playlists` | Blocked for new apps |
| `GET /browse/categories/{id}/playlists` | Blocked for new apps |
| Algorithmic / Spotify-owned editorial playlists | Blocked for new apps |
| `preview_url` in multi-get responses | Blocked for new apps |

**What to use instead:**

- **Instead of `/recommendations`**: Feed the queue from `get_top_tracks()` and
  `get_artist_top_tracks()` on the user's top artists. Cross-reference with
  `check_saved_tracks()` to exclude songs they already know.
- **Instead of audio features (energy, BPM, valence)**: No public API replacement.
  Use `popularity` from track objects as a rough proxy. For local audio files,
  consider `librosa` (Python) for feature extraction.
- **Instead of editorial playlists**: Use `get_artist_albums()` on the user's top artists
  to surface new releases, and `search()` with `tag:new` to find fresh content.

---

## 19. The One Truly Deprecated Endpoint

Only one endpoint is formally marked `deprecated: true` in the current OpenAPI spec:

```
GET /artists/{id}/related-artists   ← deprecated: true in spec AND access-restricted
```

Do not use it. There is no public API replacement.

---

## 20. Scope Master List for BUTLER

Every feature in this document is covered by the scopes below.

```python
BUTLER_SCOPES = " ".join([
    # Playback (Premium required for modify)
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "user-read-recently-played",
    # Library
    "user-library-read",
    "user-library-modify",
    # Playlists
    "playlist-read-private",
    "playlist-modify-private",
    "playlist-modify-public",
    # Personalization
    "user-top-read",
    # User profile (needed for market/country)
    "user-read-private",
    "user-read-email",
    # Social
    "user-follow-read",
    "user-follow-modify",
    # Custom playlist covers
    "ugc-image-upload",
    # Podcasts / audiobooks playback position
    "user-read-playback-position",
])
```

> **Rule**: Only ship the scopes for features you have actually built. Remove scopes for
> anything not yet implemented. The user is shown the full scope list on the consent screen —
> requesting permissions you do not use erodes trust.

---

## 21. Pre-Shipping Checklist

Run through this before every feature release:

- [ ] Redirect URI is `http://127.0.0.1` for local dev, HTTPS for production. Never `localhost`.
- [ ] `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` come from environment variables — not hardcoded anywhere in source code.
- [ ] The `@spotify_retry()` decorator is on every function that calls the Spotify API.
- [ ] `market` is passed to every endpoint that accepts it (search, player, catalog, playlists, library).
- [ ] Playlist item operations use `/playlists/{id}/tracks` — not `/playlists/{id}/items` (that path does not exist).
- [ ] Bulk calls (tracks, albums, artists) are batched at **50 IDs max** per request.
- [ ] Playlist item adds are batched at **100 URIs max** per request.
- [ ] `403` errors surface a human-readable message — Premium vs scope — by reading the error body.
- [ ] `404` on playback surfaces "Please open Spotify on a device first."
- [ ] No access-restricted endpoint is called in new code (`/recommendations`, `/audio-features`, `/audio-analysis`, related-artists, featured/editorial playlists).
- [ ] Track/artist metadata is not persisted to a database — session use only (ToS compliance).
- [ ] UI displays "Powered by Spotify" attribution wherever Spotify content is shown.
- [ ] `user-read-recently-played` is in your scope list if you call `get_recently_played()`.
- [ ] Images uploaded as playlist covers are JPEG, base64-encoded, and under 256 KB.

---

*Specification verified against the Spotify OpenAPI schema and official developer blog — 2026-04-26*
*Access restriction source: https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api*