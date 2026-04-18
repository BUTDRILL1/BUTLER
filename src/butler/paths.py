from __future__ import annotations

import os
from pathlib import Path


def butler_home_dir() -> Path:
    override = os.getenv("BUTLER_HOME")
    if override:
        p = Path(override).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Prefer AppData, but fall back to a local folder when the environment blocks writes
    # outside the workspace (common in sandboxed/dev environments).
    candidates: list[Path] = []
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        candidates.append((Path(local_app_data) / "BUTLER").resolve())
    app_data = os.getenv("APPDATA")
    if app_data:
        candidates.append((Path(app_data) / "BUTLER").resolve())
    candidates.append((Path.home() / ".butler").resolve())
    candidates.append((Path.cwd() / ".butler").resolve())

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except PermissionError:
            continue

    raise PermissionError(
        "Could not create a BUTLER data directory. Set BUTLER_HOME to a writable folder."
    )


def ensure_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Access denied creating directory: {path}. Set BUTLER_HOME to a writable folder."
        ) from e
    return path


def config_path() -> Path:
    return butler_home_dir() / "config.json"


def db_path() -> Path:
    return butler_home_dir() / "butler.sqlite3"


def notes_dir() -> Path:
    return butler_home_dir() / "notes"
