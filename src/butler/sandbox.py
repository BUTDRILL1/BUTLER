from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SandboxError(Exception):
    pass


@dataclass(frozen=True)
class PathSandbox:
    allowed_roots: tuple[Path, ...]

    @staticmethod
    def from_strings(paths: list[str]) -> "PathSandbox":
        roots: list[Path] = []
        for p in paths:
            try:
                roots.append(Path(p).expanduser().resolve())
            except FileNotFoundError:
                # resolve can raise if drive missing; treat as invalid root
                continue
        return PathSandbox(tuple(roots))

    def ensure_allowed(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise SandboxError(f"Path is outside allowed roots: {resolved}")
