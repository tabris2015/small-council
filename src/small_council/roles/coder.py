"""Coder role: implements the plan as a single fenced Python solution."""

from __future__ import annotations

from pydantic_ai import Agent

CODER_SYSTEM = (
    "You are an expert Python programmer. Implement the task following the given plan.\n"
    "Rules:\n"
    "- Define exactly the requested entrypoint with the exact name and signature.\n"
    "- Return ONLY the solution code inside one ```python code block. No explanation, no tests, "
    "no example usage.\n"
    "- Do not read from stdin or print; the entrypoint is imported and called directly.\n"
    "- Use only the Python standard library."
)

coder_agent = Agent(output_type=str, system_prompt=CODER_SYSTEM, name="coder")
