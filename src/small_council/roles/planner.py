"""Planner role: turns a task into a short implementation plan (no code)."""

from __future__ import annotations

from pydantic_ai import Agent

PLANNER_SYSTEM = (
    "You are a senior algorithm designer. Given a coding task, write a SHORT plan (3-6 bullet "
    "points) the implementer will follow: the core approach/data structure, the target time "
    "complexity, and the edge cases that must be handled. Do NOT write code — plan only."
)

planner_agent = Agent(output_type=str, system_prompt=PLANNER_SYSTEM, name="planner")
