"""Ports (protocols) the council depends on, so backends can be swapped without touching logic."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from small_council.models import ExampleReport


@runtime_checkable
class CodeRunner(Protocol):
    """Runs candidate code against worked-example expressions in an isolated sandbox.

    The default is :class:`small_council.runners.local_subprocess.LocalCodeRunner`. The bench
    integration can inject its own hardened executor instead, so the council reuses whatever sandbox
    the host already trusts.
    """

    def run_examples(self, code: str, examples: list[str]) -> ExampleReport: ...
