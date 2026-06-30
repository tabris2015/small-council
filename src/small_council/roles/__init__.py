"""The three council roles, each a Pydantic-AI agent with a fixed system prompt.

Agents are created without a bound model; the orchestrator passes the per-role model at call time
(``run_sync(..., model=...)``), which is how the bench injects its own model layer.
"""

from small_council.roles.coder import CODER_SYSTEM, coder_agent
from small_council.roles.planner import PLANNER_SYSTEM, planner_agent
from small_council.roles.verifier import VERIFIER_SYSTEM, verifier_agent

__all__ = [
    "CODER_SYSTEM",
    "PLANNER_SYSTEM",
    "VERIFIER_SYSTEM",
    "coder_agent",
    "planner_agent",
    "verifier_agent",
]
