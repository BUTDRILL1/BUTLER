from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from butler.agent.loop import AgentRuntime
from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.tools.impl.spotify_control import handle_spotify_play, SpotifyPlayArgs
from butler.tools.registry import build_default_tool_registry
from butler.voice.normalize import normalize_text
from butler.sandbox import PathSandbox
from butler.tools.base import ToolContext


def test_transcript_normalization_applies_aliases_and_drops_fillers() -> None:
    text = normalize_text(
        "Bro please play husn by anuv jane",
        aliases={"anuv jane": "Anuv Jain"},
    )
    assert text == "play husn by Anuv Jain"


@dataclass
class FakeSpotifyClient:
    start_calls: list[tuple[tuple[Any, ...], dict[str, Any]]]

    def search(self, q: str, type: str, limit: int):  # noqa: A003
        if type == "track":
            return {
                "tracks": {
                    "items": [
                        {
                            "uri": "spotify:track:1",
                            "name": "Husn",
                            "artists": [{"name": "Anuv Jain"}],
                            "popularity": 95,
                            "album": {"name": "Husn"},
                        },
                        {
                            "uri": "spotify:track:2",
                            "name": "Husn - Live",
                            "artists": [{"name": "Anuv Jain"}],
                            "popularity": 80,
                            "album": {"name": "Husn (Live)"},
                        },
                    ]
                }
            }
        if type == "artist":
            return {"artists": {"items": [{"id": "artist-1", "name": "Anuv Jain"}]}}
        return {}

    def artist_top_tracks(self, artist_id: str):
        return {
            "tracks": [
                {
                    "uri": "spotify:track:1",
                    "name": "Husn",
                    "artists": [{"name": "Anuv Jain"}],
                    "popularity": 95,
                    "album": {"name": "Husn"},
                }
            ]
        }

    def devices(self):
        return {"devices": [{"id": "device-1", "is_active": True}]}

    def start_playback(self, *args: Any, **kwargs: Any):
        self.start_calls.append((args, kwargs))


def test_spotify_play_uses_aliases_and_ranking(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    ctx = ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))
    fake_client = FakeSpotifyClient(start_calls=[])
    monkeypatch.setattr("butler.tools.impl.spotify_control._get_spotify_client", lambda ctx: fake_client)

    out = handle_spotify_play(ctx, SpotifyPlayArgs(query="play husn by anuv jane"))

    assert out["success"] is True
    assert out["track_name"] == "Husn"
    assert out["artist"] == "Anuv Jain"
    assert fake_client.start_calls


def test_spotify_play_requests_clarification_when_confidence_is_low(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    ctx = ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))

    class AmbiguousClient(FakeSpotifyClient):
        def search(self, q: str, type: str, limit: int):  # noqa: A003
            if type == "track":
                return {
                    "tracks": {
                        "items": [
                            {
                                "uri": "spotify:track:9",
                                "name": "Completely Different Song",
                                "artists": [{"name": "Different Artist"}],
                                "popularity": 20,
                                "album": {"name": "Different"},
                            }
                        ]
                    }
                }
            return {"artists": {"items": []}}

    fake_client = AmbiguousClient(start_calls=[])
    monkeypatch.setattr("butler.tools.impl.spotify_control._get_spotify_client", lambda ctx: fake_client)

    out = handle_spotify_play(ctx, SpotifyPlayArgs(query="play zzzzz"))

    assert out["success"] is True
    assert out["needs_clarification"] is True
    assert "Did you mean" in out["question"]
    assert not fake_client.start_calls


def test_agent_runtime_emits_clarification_directly(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    tools = build_default_tool_registry(cfg, db)

    class Provider:
        def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> str:
            return '{"type":"tool_call","name":"spotify.play","arguments":{"query":"play zzzzz"}}'

    runtime = AgentRuntime(config=cfg, db=db, tools=tools, provider=Provider())
    runtime.tools.call = lambda tool_name, args, conversation_id=None: {
        "success": True,
        "needs_clarification": True,
        "question": "Did you mean 'Husn' by Anuv Jain?",
    }  # type: ignore[method-assign]

    out = runtime.chat_once("play zzzzz")

    assert out == "Did you mean 'Husn' by Anuv Jain?"
