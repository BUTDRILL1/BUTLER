from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from butler.paths import config_path, ensure_dir


class ApiKeyConfig(BaseModel):
    key: str
    label: str


class ButlerConfig(BaseModel):
    assistant_name: str = "BUTLER"
    ollama_url: str = Field(default_factory=lambda: os.getenv("BUTLER_OLLAMA_URL", "http://127.0.0.1:11434"))
    provider: str = Field(default_factory=lambda: os.getenv("BUTLER_PROVIDER", "ollama"))
    gemini_api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    claude_api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    nvidia_api_keys: list[ApiKeyConfig] = Field(default_factory=list)

    # Convenience properties for the active (first) key
    @property
    def gemini_api_key(self) -> str:
        return self.gemini_api_keys[0].key if self.gemini_api_keys else ""

    @property
    def claude_api_key(self) -> str:
        return self.claude_api_keys[0].key if self.claude_api_keys else ""

    @property
    def nvidia_api_key(self) -> str:
        return self.nvidia_api_keys[0].key if self.nvidia_api_keys else ""

    @model_validator(mode="before")
    @classmethod
    def _migrate_single_keys(cls, data: dict) -> dict:
        """Backward compat: migrate old single-string or list-of-string key fields to ApiKeyConfig format."""
        if isinstance(data, dict):
            for old, new in (("gemini_api_key", "gemini_api_keys"), ("claude_api_key", "claude_api_keys"), ("nvidia_api_key", "nvidia_api_keys")):
                if old in data and new not in data:
                    val = data.pop(old)
                    data[new] = [val] if val else []
                    
            for key_field in ("gemini_api_keys", "claude_api_keys", "nvidia_api_keys"):
                if key_field in data and isinstance(data[key_field], list):
                    migrated = []
                    for item in data[key_field]:
                        if isinstance(item, str):
                            migrated.append({"key": item, "label": "default"})
                        elif isinstance(item, dict):
                            migrated.append(item)
                    data[key_field] = migrated

            def _parse_env_keys(env_str: str) -> list[dict]:
                if not env_str: return []
                env_str = env_str.strip()
                if env_str.startswith("["):
                    try:
                        import json
                        parsed = json.loads(env_str)
                        if isinstance(parsed, list):
                            return parsed
                    except:
                        pass
                return [{"key": env_str, "label": "env"}]

            # Force override from env vars to ensure .env is the source of truth
            env = os.getenv("BUTLER_GEMINI_API_KEY", "")
            if env:
                data["gemini_api_keys"] = _parse_env_keys(env)
            
            env = os.getenv("BUTLER_CLAUDE_API_KEY", "")
            if env:
                data["claude_api_keys"] = _parse_env_keys(env)
                
            env = os.getenv("BUTLER_NVIDIA_API_KEY", "")
            if env:
                data["nvidia_api_keys"] = _parse_env_keys(env)
        return data
    spotify_client_id: str = Field(default_factory=lambda: os.getenv("BUTLER_SPOTIFY_CLIENT_ID", ""))
    spotify_client_secret: str = Field(default_factory=lambda: os.getenv("BUTLER_SPOTIFY_CLIENT_SECRET", ""))
    stt_model: str = Field(default_factory=lambda: os.getenv("BUTLER_STT_MODEL", "small"))
    stt_language: str = Field(default_factory=lambda: os.getenv("BUTLER_STT_LANGUAGE", "en"))
    voice: str = Field(default_factory=lambda: os.getenv("BUTLER_VOICE", "en-IE-EmilyNeural"))
    tts_provider: str = Field(default_factory=lambda: os.getenv("BUTLER_TTS_PROVIDER", "edge-tts"))
    model: str = Field(default_factory=lambda: os.getenv("BUTLER_MODEL", "mistral:7b-instruct"))
    chat_model: str = Field(default_factory=lambda: os.getenv("BUTLER_CHAT_MODEL", "gemma:2b"))
    vision_model: str = Field(default_factory=lambda: os.getenv("BUTLER_VISION_MODEL", "google/gemma-3n-e4b-it"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("BUTLER_EMBEDDING_MODEL", "nvidia/nv-embed-v1"))
    skill_compiler_provider: str = Field(default_factory=lambda: os.getenv("BUTLER_SKILL_COMPILER_PROVIDER", os.getenv("BUTLER_PROVIDER", "ollama")))
    skill_compiler_model: str = Field(default_factory=lambda: os.getenv("BUTLER_SKILL_COMPILER_MODEL", os.getenv("BUTLER_MODEL", "mistral:7b-instruct")))
    fallback_models: list[str] = Field(default_factory=list)
    nvidia_tts_reference_audio: str = Field(default_factory=lambda: os.getenv("BUTLER_NVIDIA_TTS_REFERENCE_AUDIO", ""))
    rate: str = "+0%"
    persona: str = "Executive"
    home_location: str = ""
    user_name: str = "Boss"
    voice_prebuffer_seconds: float = Field(default_factory=lambda: float(os.getenv("BUTLER_VOICE_PREBUFFER_SECONDS", "0.75")))
    voice_silence_seconds: float = Field(default_factory=lambda: float(os.getenv("BUTLER_VOICE_SILENCE_SECONDS", "2.0")))
    voice_energy_threshold: float = Field(default_factory=lambda: float(os.getenv("BUTLER_VOICE_ENERGY_THRESHOLD", "0.01")))
    voice_wake_word_aliases: list[str] = Field(default_factory=lambda: ["butler", "batler", "budler"])
    transcript_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "noida": "Noida",
            "noyda": "Noida",
            "gurgoan": "Gurugram",
            "gurgaon": "Gurugram",
            "bangalore": "Bengaluru",
            "banglore": "Bengaluru",
            "anuv jane": "Anuv Jain",
            "anuv jain": "Anuv Jain",
            "the weekand": "The Weeknd",
        }
    )
    spotify_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "anuv jane": "Anuv Jain",
            "anuv jain": "Anuv Jain",
            "the weekand": "The Weeknd",
            "arjit singh": "Arijit Singh",
            "arijit singh": "Arijit Singh",
            "noida": "Noida",
            "noyda": "Noida",
        }
    )
    spotify_accept_confidence: float = Field(default_factory=lambda: float(os.getenv("BUTLER_SPOTIFY_ACCEPT_CONFIDENCE", "0.72")))
    spotify_clarify_confidence: float = Field(default_factory=lambda: float(os.getenv("BUTLER_SPOTIFY_CLARIFY_CONFIDENCE", "0.55")))
    spotify_search_limit: int = Field(default_factory=lambda: int(os.getenv("BUTLER_SPOTIFY_SEARCH_LIMIT", "10")))

    model_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("BUTLER_MODEL_TIMEOUT_SECONDS", "180")))
    model_retry_count: int = Field(default_factory=lambda: int(os.getenv("BUTLER_MODEL_RETRY_COUNT", "1")))
    model_total_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("BUTLER_MODEL_TOTAL_TIMEOUT_SECONDS", "220")))

    allowed_roots: list[str] = Field(default_factory=list)
    confirm_writes: bool = False
    auto_approve_tools: list[str] = Field(default_factory=list)

    blocked_web_domains: list[str] = Field(
        default_factory=lambda: [
            "facebook.com", "instagram.com", "x.com", "quora.com", "pinterest.com"
        ]
    )

    telegram_bot_token: str = Field(default_factory=lambda: os.getenv("BUTLER_TELEGRAM_BOT_TOKEN", ""))
    telegram_allowed_user: str = Field(default_factory=lambda: os.getenv("BUTLER_TELEGRAM_ALLOWED_USER", ""))
    telegram_chat_id: int | None = None

    github_token: str = Field(default_factory=lambda: os.getenv("BUTLER_GITHUB_TOKEN", ""))

    supabase_url: str = Field(default_factory=lambda: os.getenv("BUTLER_SUPABASE_URL", ""))
    supabase_key: str = Field(default_factory=lambda: os.getenv("BUTLER_SUPABASE_KEY", ""))

    max_file_bytes: int = 512_000
    tool_timeout_seconds: int = 10
    max_tool_iterations: int = 6


def load_config() -> ButlerConfig:
    from dotenv import load_dotenv
    # Load .env file from the current directory, overriding existing env vars if needed
    load_dotenv(override=True)

    path = config_path()
    ensure_dir(path.parent)
    if not path.exists():
        cfg = ButlerConfig()
        save_config(cfg)
        return cfg
    # Be tolerant of UTF-8 BOM (common on Windows when files are edited by some tools).
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw) if raw.strip() else {}
    
    # SECURITY & CONFIG MIGRATION: Force env vars to override plaintext JSON config
    for key, env_var in [
        ("spotify_client_id", "BUTLER_SPOTIFY_CLIENT_ID"),
        ("spotify_client_secret", "BUTLER_SPOTIFY_CLIENT_SECRET"),
        ("telegram_bot_token", "BUTLER_TELEGRAM_BOT_TOKEN"),
        ("telegram_allowed_user", "BUTLER_TELEGRAM_ALLOWED_USER"),
        ("model", "BUTLER_MODEL"),
        ("chat_model", "BUTLER_CHAT_MODEL"),
        ("vision_model", "BUTLER_VISION_MODEL"),
        ("embedding_model", "BUTLER_EMBEDDING_MODEL"),
        ("skill_compiler_provider", "BUTLER_SKILL_COMPILER_PROVIDER"),
        ("skill_compiler_model", "BUTLER_SKILL_COMPILER_MODEL"),
    ]:
        if os.getenv(env_var) and key in data:
            del data[key]
            
    return ButlerConfig.model_validate(data)


def save_config(config: ButlerConfig) -> None:
    path = config_path()
    ensure_dir(path.parent)
    path.write_text(json.dumps(config.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
