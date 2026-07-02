"""Tool-calling coder: a single agent with a ``run_python`` tool that self-debugs against examples.

Two modes, for the tool-reliability ablation:

- ``native``: Pydantic-AI registers ``run_python`` as a real tool; the model emits *structured*
  tool calls that ``mlx_lm.server`` parses via the model's tool-call template. Model-driven — the
  agent decides when to test and iterate.
- ``prompted``: no tool schema; the model is told to emit ``RUN`` + a fenced block that we parse and
  execute, feeding stdout/stderr back as the next turn. The unstructured baseline.

Both return a ``CouncilResult`` plus a stats dict (``tool_calls``, ``tool_errors``) so we can report
tool-call validity — the research's "tool-init reliability" signal.
"""

from __future__ import annotations

import re

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from small_council.extract import extract_code
from small_council.models import CouncilResult, CouncilTask, Usage
from small_council.runners.local_subprocess import run_snippet

NATIVE_SYSTEM = (
    "You are an expert Python programmer with a `run_python` tool that executes Python code and "
    "returns its stdout/stderr. Implement the requested entrypoint, then you MUST call `run_python` "
    "at least once to test it against the task's worked examples before answering, and fix any "
    "failure you observe. When confident it is correct, reply with ONLY the final solution inside "
    "one ```python code block."
)

PROMPTED_SYSTEM = (
    "You are an expert Python programmer. You may test code by emitting a block of exactly this "
    "form:\nRUN\n```python\n<code that prints something>\n```\nand I will reply with its "
    "stdout/stderr. Implement the requested entrypoint, test it against the task's worked "
    "examples, and fix failures. When confident, reply with the final solution in one ```python "
    "code block and no RUN block."
)

_RUN_RE = re.compile(r"RUN\s*```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _user(task: CouncilTask) -> str:
    parts = [
        task.prompt.strip(),
        "",
        f"Define a top-level Python entrypoint named `{task.entrypoint}`.",
    ]
    if task.signature:
        parts.append(f"Expected signature/usage: {task.signature}")
    return "\n".join(parts)


def _usage(u) -> Usage:
    return Usage(
        requests=u.requests or 0,
        prompt_tokens=u.input_tokens or 0,
        completion_tokens=u.output_tokens or 0,
    )


def _tool_result(r: dict) -> str:
    return (
        f"exit_ok={r['ok']} timed_out={r['timed_out']}\n"
        f"stdout:\n{r['stdout']}\nstderr:\n{r['stderr']}"
    )[:2500]


def run_tool_agent_native(
    task: CouncilTask,
    *,
    model: Model,
    max_calls: int = 6,
    timeout_s: float = 10.0,
    temperature: float = 0.0,
    max_tokens: int = 3072,
) -> tuple[CouncilResult, dict]:
    stats = {"mode": "native", "tool_calls": 0, "tool_errors": 0}
    agent = Agent(model=model, system_prompt=NATIVE_SYSTEM, output_type=str, retries=2)

    @agent.tool_plain
    def run_python(code: str) -> str:  # noqa: D401 - tool
        stats["tool_calls"] += 1
        return _tool_result(run_snippet(code, timeout_s=timeout_s))

    settings = ModelSettings(temperature=temperature, max_tokens=max_tokens)
    raw, usage = "", Usage()
    try:
        res = agent.run_sync(
            _user(task),
            model_settings=settings,
            usage_limits=UsageLimits(request_limit=max_calls + 3),
        )
        raw, usage = res.output, _usage(res.usage)
    except Exception:
        stats["tool_errors"] += 1  # a tool-call the model/endpoint could not complete
    code, ok = extract_code(raw, entrypoint=task.entrypoint)
    return CouncilResult(
        code=code, raw_output=raw, extraction_ok=ok, attempts=stats["tool_calls"], usage=usage
    ), stats


def run_tool_agent_prompted(
    task: CouncilTask,
    *,
    model: Model,
    max_calls: int = 6,
    timeout_s: float = 10.0,
    temperature: float = 0.0,
    max_tokens: int = 3072,
) -> tuple[CouncilResult, dict]:
    stats = {"mode": "prompted", "tool_calls": 0, "tool_errors": 0}
    agent = Agent(model=model, system_prompt=PROMPTED_SYSTEM, output_type=str, retries=1)
    settings = ModelSettings(temperature=temperature, max_tokens=max_tokens)
    history = None
    prompt = _user(task)
    total, raw = Usage(), ""
    for _ in range(max_calls + 1):
        try:
            res = agent.run_sync(prompt, message_history=history, model_settings=settings)
        except Exception:
            break
        raw = res.output
        total = total + _usage(res.usage)
        history = res.all_messages()
        m = _RUN_RE.search(raw)
        if not m:
            if "RUN" in raw:  # tried to invoke the tool but emitted a malformed block
                stats["tool_errors"] += 1
            break
        stats["tool_calls"] += 1
        feedback = _tool_result(run_snippet(m.group(1), timeout_s=timeout_s))
        prompt = "Result of your RUN block:\n" + feedback + "\nContinue, or give the FINAL solution."
    code, ok = extract_code(raw, entrypoint=task.entrypoint)
    return CouncilResult(
        code=code, raw_output=raw, extraction_ok=ok, attempts=stats["tool_calls"], usage=total
    ), stats
