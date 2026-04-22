from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import random
import sqlite3
import shlex
import sys
import threading
import itertools
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_tts import list_voices

from butler.agent.loop import AgentRuntime
from butler.config import ButlerConfig, load_config, save_config
from butler.cli_menu import MenuItem, mask_secret, prompt_text, select_menu
from butler.db import ButlerDB
from butler.paths import butler_home_dir
from butler.sandbox import PathSandbox
from butler.tools.registry import build_default_tool_registry


@dataclass(frozen=True)
class CliEnv:
    stdin_is_tty: bool
    stdout_is_tty: bool


class Spinner:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        for c in itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]):
            if self._stop.is_set():
                break
            print(f"\r  {c} Thinking...", end="", flush=True)
            time.sleep(0.08)
        print("\r" + " " * 20 + "\r", end="", flush=True)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()


def _confirm(prompt: str, assume_no: bool = True) -> bool:
    suffix = " [y/N] " if assume_no else " [Y/n] "
    ans = input(prompt + suffix).strip().lower()
    if not ans:
        return not assume_no
    return ans in ("y", "yes")


def _print_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _manage_api_keys(config: ButlerConfig, provider_name: str, keys_field: str) -> ButlerConfig:
    """Interactive loop: show existing keys, add one at a time, detect duplicates."""
    from butler.config import ApiKeyConfig
    keys: list[ApiKeyConfig] = list(getattr(config, keys_field))
    while True:
        print(f"\n--- {provider_name} API Keys ({len(keys)} loaded) ---")
        for i, k in enumerate(keys, 1):
            print(f"  {i}. {mask_secret(k.key)} ({k.label})")
        if not keys:
            print("  (none)")

        new_key = input("\nEnter new key (or press Enter to go back): ").strip()
        if not new_key:
            break
        if any(k.key == new_key for k in keys):
            print("  ⚠ Key already exists. Skipped.")
        else:
            while True:
                new_label = input("Enter a label/comment for this key (mandatory): ").strip()
                if new_label:
                    break
                print("  ⚠ Label cannot be empty. Please provide a comment.")
            keys.append(ApiKeyConfig(key=new_key, label=new_label))
            print("  ✓ Key added.")

        more = input("Add another? [y/N]: ").strip().lower()
        if more not in ("y", "yes"):
            break

    new_config = config.model_copy(update={keys_field: keys})
    save_config(new_config)
    return new_config


def _changeable_config_items(config: ButlerConfig) -> list[MenuItem[str]]:
    return [
        MenuItem("Gemini API Keys", "gemini_api_keys", f"{len(config.gemini_api_keys)} keys loaded"),
        MenuItem("Claude API Keys", "claude_api_keys", f"{len(config.claude_api_keys)} keys loaded"),
        MenuItem("Nvidia API Keys", "nvidia_api_keys", f"{len(config.nvidia_api_keys)} keys loaded"),
        MenuItem("Spotify Client ID", "spotify_client_id", f"Current: {mask_secret(config.spotify_client_id)}"),
        MenuItem("Spotify Client Secret", "spotify_client_secret", f"Current: {mask_secret(config.spotify_client_secret)}"),
        MenuItem("STT Model", "stt_model", f"Current: {config.stt_model}"),
        MenuItem("STT Language", "stt_language", f"Current: {config.stt_language}"),
        MenuItem("Default Voice", "voice", f"Current: {config.voice}"),
        MenuItem("Fallback Models", "fallback_models", f"Current: {', '.join(config.fallback_models) or 'none'}"),
    ]



def _load_edge_voices() -> list[dict[str, Any]]:
    voices = asyncio.run(list_voices())
    voices.sort(key=lambda v: (v.get("Locale", ""), v.get("Gender", ""), v.get("ShortName", "")))
    return voices


def _select_voice(config: ButlerConfig) -> ButlerConfig:
    try:
        voices = _load_edge_voices()
    except Exception as e:  # noqa: BLE001
        print(f"Could not load voices: {e}")
        return config

    items = [
        MenuItem(
            label=f"{voice.get('ShortName', 'Unknown')} ({voice.get('Locale', '')}, {voice.get('Gender', '')})",
            value=voice,
            description=voice.get("FriendlyName", ""),
        )
        for voice in voices
    ]
    selected = select_menu("Choose a default TTS voice", items, page_size=12)
    if selected is None:
        return config

    voice_name = selected.value.get("ShortName", config.voice)
    new_config = config.model_copy(update={"voice": voice_name})
    save_config(new_config)
    print(f"Saved voice: {voice_name}")
    return new_config


def _change_config_value(config: ButlerConfig) -> ButlerConfig:
    while True:
        selected = select_menu("\n--- BUTLER Configuration ---\nSelect provider to configure", _changeable_config_items(config), page_size=8)
        if selected is None:
            return config

        key = selected.value
        if key == "voice":
            config = _select_voice(config)
            continue
        if key in ("gemini_api_keys", "claude_api_keys", "nvidia_api_keys"):
            label = "Gemini" if "gemini" in key else ("Claude" if "claude" in key else "Nvidia")
            config = _manage_api_keys(config, label, key)
            continue

        secret_fields = {"spotify_client_id", "spotify_client_secret"}
        
        if key == "fallback_models":
            current_value = ", ".join(getattr(config, key))
            new_value_str = prompt_text("Fallback Models (comma separated)", current=current_value)
            if new_value_str is None:
                continue
            new_value = [m.strip() for m in new_value_str.split(",") if m.strip()]
        else:
            current_value = getattr(config, key)
            new_value = prompt_text(selected.label, current=str(current_value), secret=key in secret_fields)
            if new_value is None:
                continue

        config = config.model_copy(update={key: new_value})
        save_config(config)
        print(f"Saved {selected.label}.")


def _cmd_roots(config: ButlerConfig, args: list[str]) -> ButlerConfig:
    parser = argparse.ArgumentParser(prog="/roots", add_help=False)
    sub = parser.add_subparsers(dest="sub", required=True)
    p_add = sub.add_parser("add")
    p_add.add_argument("path")
    p_delete = sub.add_parser("delete")
    p_delete.add_argument("path")
    sub.add_parser("list")
    ns = parser.parse_args(args)

    if ns.sub == "list":
        if not config.allowed_roots:
            print("(no roots configured)")
        for p in config.allowed_roots:
            print(p)
        return config

    if ns.sub == "delete":
        raw = Path(ns.path).expanduser()
        resolved = str(raw.resolve())
        roots = [r for r in config.allowed_roots if r != resolved]
        new_config = config.model_copy(update={"allowed_roots": list(dict.fromkeys(roots))})
        save_config(new_config)
        print(f"Deleted root: {resolved}")
        return new_config

    raw = Path(ns.path).expanduser()
    resolved = str(raw.resolve())
    roots = list(dict.fromkeys([*config.allowed_roots, resolved]))
    new_config = config.model_copy(update={"allowed_roots": roots})
    save_config(new_config)
    print(f"Added root: {resolved}")
    return new_config


def _cmd_index(config: ButlerConfig, runtime: AgentRuntime, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="/index", add_help=False)
    sub = parser.add_subparsers(dest="sub", required=True)
    sub.add_parser("sync")
    ns = parser.parse_args(args)
    if ns.sub == "sync":
        result = runtime.tools.call("index.sync", {"max_files": 2000})
        _print_json(result)


def _cmd_notes(config: ButlerConfig, runtime: AgentRuntime, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="/notes", add_help=False)
    sub = parser.add_subparsers(dest="sub", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("title")
    p_create.add_argument("content")

    p_append = sub.add_parser("append")
    p_append.add_argument("title")
    p_append.add_argument("content")

    p_read = sub.add_parser("read")
    p_read.add_argument("title")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")

    ns = parser.parse_args(args)

    if ns.sub == "create":
        if config.confirm_writes and not _confirm(f"Create note '{ns.title}'?"):
            print("Canceled.")
            return
        _print_json(runtime.tools.call("notes.create", {"title": ns.title, "content": ns.content}))
        return
    if ns.sub == "append":
        if config.confirm_writes and not _confirm(f"Append to note '{ns.title}'?"):
            print("Canceled.")
            return
        _print_json(runtime.tools.call("notes.append", {"title": ns.title, "content": ns.content}))
        return
    if ns.sub == "read":
        _print_json(runtime.tools.call("notes.read", {"title": ns.title}))
        return
    if ns.sub == "search":
        _print_json(runtime.tools.call("notes.search", {"query": ns.query}))
        return


def _handle_slash_command(
    line: str, config: ButlerConfig, runtime: AgentRuntime
) -> ButlerConfig:
    parts = shlex.split(line, posix=False)
    cmd = parts[0]
    args = parts[1:]
    if cmd == "/roots":
        return _cmd_roots(config, args)
    if cmd == "/index":
        _cmd_index(config, runtime, args)
        return config
    if cmd == "/notes":
        _cmd_notes(config, runtime, args)
        return config
    if cmd == "/config":
        _print_json(config.model_dump())
        return config
    if cmd == "/new":
        runtime.conversation_id = runtime.db.new_conversation()
        print("Started a fresh conversation session.")
        return config
    if cmd == "/help":
        print("Commands:")
        print("  /roots add <path> | /roots delete <path> | /roots list")
        print("  /index sync")
        print('  /notes create "title" "content"')
        print('  /notes append "title" "content"')
        print('  /notes read "title"')
        print('  /notes search "query"')
        print("  /new")
        print("  /config")
        print("  /exit")
        return config
    if cmd == "/exit":
        raise SystemExit(0)
    print(f"Unknown command: {cmd}. Try /help.")
    return config


_BOOT_GREETINGS = [
    "Hey Boss! BUTLER's rebooting!",
    "Yo Boss! BUTLER's back online!",
    "Boss! BUTLER warming up!",
    "Rise and grind, Boss! BUTLER's loading!",
    "BUTLER powering up, Boss!",
]


def main(argv: list[str] | None = None) -> int:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="butler")
    parser.add_argument("--chat", nargs="*", help="One-shot chat prompt")
    parser.add_argument("--gem", "--gemini", action="store_true", help="Use Gemini API instead of local Ollama")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical floating widget")
    parser.add_argument("--claude", action="store_true", help="Use Anthropic Claude API instead of local Ollama")
    parser.add_argument("--nvidia", "--nim", action="store_true", help="Use Nvidia NIM API instead of local Ollama")
    parser.add_argument("--change", action="store_true", help="Change configured keys and settings")
    parser.add_argument("--voice", action="store_true", help="Pick and save a default TTS voice")
    parser.add_argument("--auth-spotify", action="store_true", help="Authenticate Spotify safely before launching the voice loop")
    ns = parser.parse_args(argv)

    log_level = os.getenv("BUTLER_LOG_LEVEL", "").strip().upper()
    if log_level:
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(levelname)s %(name)s: %(message)s")

    config = load_config()
    from butler.config import save_config
    
    if ns.change:
        config = _change_config_value(config)

    if ns.voice:
        config = _select_voice(config)

    if ns.change or ns.voice:
        return 0

    if ns.auth_spotify:
        if not config.spotify_client_id or not config.spotify_client_secret:
            print("Error: Please run 'butler --change' first to set your Spotify Client ID and Secret.")
            return 1
            
        print("\n--- Initializing Spotify Authentication ---")
        print("A browser window will now open. Click 'Agree'.")
        print("When redirected to a 'Site cannot be reached' page, copy the ENTIRE URL (https://localhost:8080/?code=...).")
        print("Paste that URL below and hit Enter.\n")
        
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        
        auth_manager = SpotifyOAuth(
            client_id=config.spotify_client_id,
            client_secret=config.spotify_client_secret,
            redirect_uri="http://127.0.0.1:8080",
            scope="user-read-playback-state user-modify-playback-state user-read-currently-playing",
            open_browser=True
        )
        
        # This will forcefully trigger the flow and generate the .cache file
        sp = spotipy.Spotify(auth_manager=auth_manager)
        sp.current_user() # Test the connection
        print("\nSuccessfully linked Spotify account! The token has been cached. You can now use voice commands.")
        return 0
    
    if ns.gem:
        config = config.model_copy(update={
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "chat_model": "gemini-2.5-flash"
        })
        
    if ns.claude:
        config = config.model_copy(update={
            "provider": "claude",
            "model": "claude-3-5-sonnet-20241022",
            "chat_model": "claude-3-5-sonnet-20241022"
        })

    if ns.nvidia:
        config = config.model_copy(update={
            "provider": "nvidia",
            "model": "mistralai/mistral-nemotron",
            "chat_model": "mistralai/mistral-nemotron"
        })

    from butler.config import save_config, ApiKeyConfig
        
    if config.provider == "gemini" and not config.gemini_api_keys:
        print("Gemini API key is required but not found in configuration.")
        key = input("Please enter your Gemini API key: ").strip()
        if not key:
            print("Error: API key cannot be empty.")
            return 2
        config = config.model_copy(update={"gemini_api_keys": [ApiKeyConfig(key=key, label="default")]})
        # Save to config to persist
        save_config(config)

    if config.provider == "claude" and not config.claude_api_keys:
        print("Anthropic Claude API key is required but not found in configuration.")
        key = input("Please enter your Anthropic API key: ").strip()
        if not key:
            print("Error: API key cannot be empty.")
            return 2
        config = config.model_copy(update={"claude_api_keys": [ApiKeyConfig(key=key, label="default")]})
        # Save to config to persist
        save_config(config)

    if config.provider == "nvidia" and not config.nvidia_api_keys:
        print("Nvidia NIM API key is required but not found in configuration.")
        key = input("Please enter your Nvidia API key (starts with nvapi-): ").strip()
        if not key:
            print("Error: API key cannot be empty.")
            return 2
        config = config.model_copy(update={"nvidia_api_keys": [ApiKeyConfig(key=key, label="default")]})
        # Save to config to persist
        save_config(config)

    db = ButlerDB.open(config)
    tools = build_default_tool_registry(config, db)
    try:
        runtime = AgentRuntime(
            config=config,
            db=db,
            tools=tools,
            confirm_tool=lambda name, arguments: _confirm(
                f"Allow tool '{name}' with args {json.dumps(arguments, ensure_ascii=False)}?"
            ),
        )
    except sqlite3.OperationalError as e:
        if "readonly" in str(e).lower():
            home = butler_home_dir()
            print(
                f"BUTLER startup error: database is read-only. Active BUTLER_HOME: {home}. "
                "Set BUTLER_HOME to a writable folder."
            )
            return 2
        raise

    if ns.gui:
        import threading
        from butler.gui import ButlerWidget
        from butler.voice.engine import VoiceEngine

        print(random.choice(_BOOT_GREETINGS))

        def on_stt_command(text: str):
            print(f"Heard: {text}")
            state.widget.after(0, lambda: state.widget.set_status("Thinking..."))
            try:
                response_tokens = list(runtime.chat_once_stream(text))
                final_text = "".join(response_tokens).strip()
                print(f"Response: {final_text}")
                if final_text:
                    engine.play_tts(final_text)
                else:
                    state.widget.after(0, lambda: state.widget.set_status("Ready"))
            except Exception as e:
                print(f"STT Error: {e}")
                state.widget.after(0, lambda: state.widget.set_status("Error"))
                
        # To avoid circular reference, initialize widget without callbacks first, or use a setup pattern
        class AppState:
            pass
        state = AppState()
        
        def safe_status(s):
            if hasattr(state, "widget"):
                state.widget.after(0, lambda: state.widget.set_status(s))
                
        engine = VoiceEngine(on_command_callback=on_stt_command, status_callback=safe_status, config=config)
        print("At your service, Boss.")
        engine.play_tts("At your service, Boss.")
        state.widget = ButlerWidget(on_mic_click=engine.trigger_manual_listen, 
                                    on_wake_toggle=engine.toggle_wake_word)
        state.widget.mainloop()
        return 0

    if ns.chat is not None:
        prompt = " ".join(ns.chat).strip()
        if not prompt:
            return 2
        for token in runtime.chat_once_stream(prompt):
            print(token, end="", flush=True)
        print()
        return 0

    env = CliEnv(stdin_is_tty=sys.stdin.isatty(), stdout_is_tty=sys.stdout.isatty())
    if env.stdin_is_tty:
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        print(f"{greeting}, Boss. BUTLER online. Type /help for commands.")

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            config = _handle_slash_command(line, config, runtime)
            # refresh runtime config/tool sandbox after config mutations
            runtime.config = config
            runtime.tools.config = config
            runtime.tools.sandbox = PathSandbox.from_strings(config.allowed_roots)
            continue
            
        spinner = Spinner()
        spinner.start()
        try:
            first = True
            for token in runtime.chat_once_stream(line):
                if first:
                    spinner.stop()
                    print("BUTLER> ", end="", flush=True)
                    first = False
                print(token, end="", flush=True)
            print()
        finally:
            spinner.stop()


if __name__ == "__main__":
    raise SystemExit(main())
