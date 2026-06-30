"""Typed I/O contracts for the council and its roles.

These are the stable public types other modules (roles, orchestrator, the bench integration) build
on. Keeping them in one place makes the agent boundaries explicit and self-documenting.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CouncilTask(BaseModel):
    """One coding task for the council.

    ``examples`` are worked ``entrypoint(...) == expected`` assertions the verifier may run in a
    sandbox; if ``None`` they are extracted from the prompt's fenced code blocks.
    """

    prompt: str
    entrypoint: str
    signature: str | None = None
    examples: list[str] | None = None


class Plan(BaseModel):
    """The planner's output: a short implementation plan, no code."""

    text: str


class Verdict(BaseModel):
    """The verifier's structured judgment of a candidate solution.

    Replaces the prototype's brittle APPROVE/REVISE regex parsing with a typed result. ``reason``
    carries the concrete bug / missing case when the decision is ``REVISE``.
    """

    decision: Literal["APPROVE", "REVISE"]
    reason: str = ""

    @property
    def approved(self) -> bool:
        return self.decision == "APPROVE"


class ExampleResult(BaseModel):
    """Outcome of evaluating one worked-example expression against a candidate."""

    expr: str
    ok: bool
    error: str | None = None


class ExampleReport(BaseModel):
    """Result of running a candidate against a task's worked examples in a sandbox."""

    n: int = 0
    loaded: bool = True
    load_error: str | None = None
    timed_out: bool = False
    results: list[ExampleResult] = Field(default_factory=list)

    @property
    def failures(self) -> list[ExampleResult]:
        return [r for r in self.results if not r.ok]

    @property
    def ok(self) -> bool:
        """True only if the candidate loaded, did not time out, and passed every example."""
        return self.loaded and not self.timed_out and not self.failures


class Usage(BaseModel):
    """Token usage accumulated across a council run's model calls."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            requests=self.requests + other.requests,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class CouncilResult(BaseModel):
    """The full outcome of running the council on one task."""

    code: str
    raw_output: str
    extraction_ok: bool
    approved: bool = False
    attempts: int = 0
    plan: str = ""
    usage: Usage = Field(default_factory=Usage)
