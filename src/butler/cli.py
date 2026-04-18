from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import shlex
import sys
import threading
import itertools
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from butler.agent.loop import AgentRuntime
from butler.config import ButlerConfig, load_config, save_config
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
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        for c in itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]):
            if not self._running:
                break
            print(f"\r  {c} Thinking...", end="", flush=True)
            time.sleep(0.08)
        print("\r" + " " * 20 + "\r", end="", flush=True)

    def stop(self):
        self._running = False
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


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="butler")
    parser.add_argument("--chat", nargs="*", help="One-shot chat prompt")
    ns = parser.parse_args(argv)

    log_level = os.getenv("BUTLER_LOG_LEVEL", "").strip().upper()
    if log_level:
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(levelname)s %(name)s: %(message)s")

    config = load_config()
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

    if ns.chat is not None:
        prompt = " ".join(ns.chat).strip()
        if not prompt:
            return 2
        response = runtime.chat_once(prompt)
        print(response)
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
        print("BUTLER> ", end="", flush=True)
        try:
            first = True
            for token in runtime.chat_once_stream(line):
                if first:
                    spinner.stop()
                    first = False
                print(token, end="", flush=True)
            print()
        finally:
            spinner.stop()


if __name__ == "__main__":
    raise SystemExit(main())
