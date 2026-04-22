from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

_RE_NON_WORD = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_RE_SPACES = re.compile(r"\s+")
_DEFAULT_FILLERS = {
    "please",
    "bro",
    "boss",
    "hey",
    "hi",
    "hello",
    "kindly",
    "just",
}


def normalize_text(text: str, *, aliases: dict[str, str] | None = None, fillers: Iterable[str] | None = None) -> str:
    cleaned = _RE_NON_WORD.sub(" ", text.casefold())
    cleaned = _RE_SPACES.sub(" ", cleaned).strip()
    if not cleaned:
        return ""

    filler_set = set(_DEFAULT_FILLERS)
    if fillers is not None:
        filler_set.update(word.casefold() for word in fillers)

    tokens = [token for token in cleaned.split() if token not in filler_set]
    normalized = " ".join(tokens).strip()
    if not normalized:
        return ""

    if aliases:
        normalized = apply_aliases(normalized, aliases)
    return _RE_SPACES.sub(" ", normalized).strip()


def apply_aliases(text: str, aliases: dict[str, str]) -> str:
    normalized = text
    for source in sorted(aliases, key=len, reverse=True):
        target = aliases[source]
        pattern = re.compile(rf"(?<!\w){re.escape(source.casefold())}(?!\w)", re.IGNORECASE)
        normalized = pattern.sub(target, normalized)
    return normalized


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


def token_overlap(a: str, b: str) -> float:
    left = set(normalize_text(a).split())
    right = set(normalize_text(b).split())
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
