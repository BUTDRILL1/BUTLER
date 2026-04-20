from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from butler.paths import config_path, ensure_dir


class ButlerConfig(BaseModel):
    assistant_name: str = "BUTLER"
    ollama_url: str = Field(default_factory=lambda: os.getenv("BUTLER_OLLAMA_URL", "http://127.0.0.1:11434"))
    provider: str = Field(default_factory=lambda: os.getenv("BUTLER_PROVIDER", "ollama"))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("BUTLER_GEMINI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("BUTLER_MODEL", "mistral:7b-instruct"))
    chat_model: str = Field(default_factory=lambda: os.getenv("BUTLER_CHAT_MODEL", "gemma:2b"))
    home_location: str = ""

    model_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("BUTLER_MODEL_TIMEOUT_SECONDS", "180")))
    model_retry_count: int = Field(default_factory=lambda: int(os.getenv("BUTLER_MODEL_RETRY_COUNT", "1")))
    model_total_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("BUTLER_MODEL_TOTAL_TIMEOUT_SECONDS", "220")))

    allowed_roots: list[str] = Field(default_factory=list)
    confirm_writes: bool = True

    blocked_web_domains: list[str] = Field(
        default_factory=lambda: [
            "facebook.com", "instagram.com", "x.com", "quora.com", "pinterest.com"
        ]
    )

    max_file_bytes: int = 512_000
    tool_timeout_seconds: int = 10
    max_tool_iterations: int = 3


def load_config() -> ButlerConfig:
    path = config_path()
    ensure_dir(path.parent)
    if not path.exists():
        cfg = ButlerConfig()
        save_config(cfg)
        return cfg
    # Be tolerant of UTF-8 BOM (common on Windows when files are edited by some tools).
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw) if raw.strip() else {}
    return ButlerConfig.model_validate(data)


def save_config(config: ButlerConfig) -> None:
    path = config_path()
    ensure_dir(path.parent)
    path.write_text(json.dumps(config.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
