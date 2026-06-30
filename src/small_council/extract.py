"""Extract a runnable Python solution from a model's (often chatty) chat output.

Vendored from slm-coding-bench so the council core stays harness-agnostic. Strategy, in order:
1. strip reasoning blocks (e.g. ``<think>...</think>`` from reasoning models);
2. prefer fenced ```python blocks; among candidates, choose one that parses and (if known)
   defines the expected entrypoint, preferring the longest;
3. fall back to any fenced block, then to the whole text;
4. as a last resort, keep the largest suffix starting at the first import/def/class that parses.

Returns ``(code, ok)`` where ``ok`` means syntactically valid Python was found.
"""

from __future__ import annotations

import ast
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\n(?P<body>.*?)```", re.DOTALL)


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except (SyntaxError, ValueError):
        return False


def _defines(src: str, entrypoint: str) -> bool:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == entrypoint:
                return True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == entrypoint:
                    return True
    return False


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    # Unclosed <think> (truncated reasoning): drop everything up to the last </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group("lang").lower(), m.group("body")) for m in _FENCE_RE.finditer(text)]


def _largest_parseable_suffix(text: str) -> str | None:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines)
              if re.match(r"^\s*(import |from |def |class |@)", ln)]
    for i in starts:
        candidate = "\n".join(lines[i:])
        if _parses(candidate):
            return candidate
    return None


def extract_code(text: str, entrypoint: str | None = None) -> tuple[str, bool]:
    cleaned = _strip_think(text)
    blocks = _fenced_blocks(cleaned)

    python_blocks = [b for lang, b in blocks if lang in ("python", "py", "python3")]
    other_blocks = [b for lang, b in blocks if lang not in ("python", "py", "python3")]

    ranked: list[str] = []
    ranked += sorted(python_blocks, key=len, reverse=True)
    ranked += sorted(other_blocks, key=len, reverse=True)

    parseable = [b for b in ranked if _parses(b)]
    if entrypoint:
        for b in parseable:
            if _defines(b, entrypoint):
                return b.strip() + "\n", True
    if parseable:
        return parseable[0].strip() + "\n", True

    if _parses(cleaned):
        return cleaned.strip() + "\n", True
    suffix = _largest_parseable_suffix(cleaned)
    if suffix is not None:
        return suffix.strip() + "\n", True

    best_effort = (ranked[0] if ranked else cleaned).strip()
    return best_effort + "\n", False
