"""Verifier role: reviews a candidate and returns a typed APPROVE/REVISE verdict.

Uses ``PromptedOutput`` (JSON-in-text, validated and retried) rather than tool-call output, so it
works against plain OpenAI-compatible chat servers like ``mlx_lm.server`` that do not implement
function-calling or guided JSON.
"""

from __future__ import annotations

from pydantic_ai import Agent, PromptedOutput

from small_council.models import Verdict

VERIFIER_SYSTEM = (
    "You are a meticulous code reviewer. You are given a coding task, a candidate Python solution, "
    "and (when available) the result of running it against the task's worked examples. "
    "Decide whether the solution is correct and robust, including the stated edge cases. "
    "Choose APPROVE if it is correct; otherwise choose REVISE and give, as the reason, the "
    "single most important concrete bug or missing case the implementer must fix. "
    "Do not write the corrected code."
)

verifier_agent = Agent(
    output_type=PromptedOutput(Verdict),
    system_prompt=VERIFIER_SYSTEM,
    name="verifier",
    retries=2,  # reprompt if a small model emits invalid JSON for the verdict
)
