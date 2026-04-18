from __future__ import annotations

from pathlib import Path

import pytest

from butler.sandbox import PathSandbox, SandboxError


def test_sandbox_allows_within_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    f = root / "a.txt"
    f.write_text("x", encoding="utf-8")

    sb = PathSandbox.from_strings([str(root)])
    assert sb.ensure_allowed(f) == f.resolve()


def test_sandbox_blocks_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("no", encoding="utf-8")

    sb = PathSandbox.from_strings([str(root)])
    with pytest.raises(SandboxError):
        sb.ensure_allowed(outside)
