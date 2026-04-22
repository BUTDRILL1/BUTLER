from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class MenuItem(Generic[T]):
    label: str
    value: T
    description: str = ""


def mask_secret(value: str, visible: int = 4) -> str:
    if not value:
        return "None"
    if len(value) <= visible:
        return "*" * len(value)
    return value[:visible] + "..." + "*" * max(0, len(value) - visible - 3)


def prompt_text(prompt: str, *, current: str = "", secret: bool = False) -> str | None:
    if secret:
        value = input(f"{prompt} [{mask_secret(current)}] (paste allowed): ").strip()
    else:
        value = input(f"{prompt} [{current or 'None'}]: ").strip()
    return value or None


def select_menu(title: str, items: Sequence[MenuItem[T]], *, page_size: int = 10) -> MenuItem[T] | None:
    if not items:
        print(f"{title}: no items available.")
        return None

    if os.name != "nt" or not sys.stdin.isatty() or not sys.stdout.isatty():
        print(title)
        for idx, item in enumerate(items, start=1):
            detail = f" - {item.description}" if item.description else ""
            print(f"{idx}. {item.label}{detail}")
        choice = input("Select a number (Enter to cancel): ").strip()
        if not choice:
            return None
        try:
            selected = int(choice) - 1
        except ValueError:
            return None
        if 0 <= selected < len(items):
            return items[selected]
        return None

    try:
        import msvcrt
    except ImportError:
        return select_menu(title, items, page_size=page_size)

    index = 0
    page_size = max(5, page_size)

    def render() -> None:
        os.system("cls")
        print(title)
        print("Use Up/Down and Enter. Esc cancels.\n")
        start = max(0, index - page_size // 2)
        end = min(len(items), start + page_size)
        start = max(0, end - page_size)
        for idx in range(start, end):
            prefix = ">" if idx == index else " "
            item = items[idx]
            detail = f" - {item.description}" if item.description else ""
            print(f"{prefix} {item.label}{detail}")

    def read_key() -> str:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return ch + msvcrt.getwch()
        return ch

    render()
    while True:
        key = read_key()
        if key in ("\r", "\n"):
            os.system("cls")
            return items[index]
        if key == "\x1b":
            os.system("cls")
            return None
        if key == "\xe0H":
            index = (index - 1) % len(items)
            render()
            continue
        if key == "\xe0P":
            index = (index + 1) % len(items)
            render()
            continue
        if key == "\xe0G":
            index = 0
            render()
            continue
        if key == "\xe0O":
            index = len(items) - 1
            render()
            continue
        if key == "\xe0I":
            index = max(0, index - page_size)
            render()
            continue
        if key == "\xe0Q":
            index = min(len(items) - 1, index + page_size)
            render()
            continue
