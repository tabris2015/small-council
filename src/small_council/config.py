"""Configuration for a council run."""

from __future__ import annotations

from pydantic import BaseModel


class CouncilConfig(BaseModel):
    """Per-role model handles plus loop and sandbox settings.

    Model handles are provider-qualified ids understood by the injected model layer
    (e.g. ``"mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"``). Any role left ``None`` falls back to
    the coder model, or to a single shared model supplied at call time.
    """

    planner_model: str | None = None
    coder_model: str | None = None
    verifier_model: str | None = None

    max_retries: int = 2
    example_timeout_s: float = 10.0
    example_mem_limit_mb: int | None = 1024

    temperature: float = 0.0
    max_tokens: int = 2048

    # For standalone use against an OpenAI-compatible endpoint. The bench integration injects its
    # own model layer and ignores these.
    base_url: str | None = None
    api_key: str | None = None
