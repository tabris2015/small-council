"""The council orchestrator: a planner -> coder -> verifier loop with bounded retries.

Promoted from the framework-less prototype in slm-coding-bench, now built on Pydantic-AI agents with
an injected per-role model and a typed verifier verdict. The control flow is unchanged and
deliberately simple (orchestrator-driven, no autonomous tool-calling yet):

    1. Planner produces a short plan.
    2. Coder implements it; the candidate is run against the task's worked examples in a sandbox.
    3. If examples fail, feed the failures back to the coder (bounded retries).
    4. If examples pass (or none exist), the verifier reviews; APPROVE ends the loop, REVISE feeds
       the reason back to the coder.

**Honesty:** the verifier only ever sees the task's worked ``## Examples`` (information every solver
gets), never a hidden graded test suite — so a verify->retry loop cannot contaminate a benchmark.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from small_council.config import CouncilConfig
from small_council.extract import extract_code, extract_examples
from small_council.models import CouncilResult, CouncilTask, ExampleReport, Usage, Verdict
from small_council.ports import CodeRunner
from small_council.roles import coder_agent, planner_agent, verifier_agent
from small_council.runners import LocalCodeRunner


class Council:
    """Runs the planner -> coder -> verifier loop over injected per-role models."""

    def __init__(
        self,
        *,
        planner_model: Model,
        coder_model: Model,
        verifier_model: Model,
        code_runner: CodeRunner | None = None,
        max_retries: int = 2,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.verifier_model = verifier_model
        self.code_runner = code_runner or LocalCodeRunner()
        self.max_retries = max_retries
        self.model_settings = model_settings

    def solve(self, task: CouncilTask) -> CouncilResult:
        examples = (
            task.examples
            if task.examples is not None
            else extract_examples(task.prompt, task.entrypoint)
        )
        usage = Usage()

        plan_res = planner_agent.run_sync(
            _planner_user(task), model=self.planner_model, model_settings=self.model_settings
        )
        plan = plan_res.output
        usage += _usage(plan_res)

        feedback: str | None = None
        code, raw, ok, approved, attempts = "", "", False, False, 0

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            code_res = coder_agent.run_sync(
                _coder_user(task, plan, feedback),
                model=self.coder_model,
                model_settings=self.model_settings,
            )
            raw = code_res.output
            usage += _usage(code_res)
            code, ok = extract_code(raw, entrypoint=task.entrypoint)

            if not ok:
                feedback = (
                    "Your previous reply did not contain a parseable Python solution inside one "
                    "```python code block. Return only the code."
                )
                continue

            report = self.code_runner.run_examples(code, examples) if examples else None
            if report is not None and report.failures:
                if attempt < self.max_retries:
                    feedback = _example_feedback(report)
                    continue
                break  # out of retries; return the last candidate for honest grading

            # Examples pass (or none): an LLM review is the remaining gate.
            verdict = self._review(task, code, report)
            usage += verdict.usage
            if verdict.value is None:
                break  # verifier could not produce a structured verdict; keep the candidate
            if verdict.value.approved or attempt == self.max_retries:
                approved = verdict.value.approved
                break
            feedback = f"A reviewer flagged a problem: {verdict.value.reason}"

        return CouncilResult(
            code=code,
            raw_output=raw,
            extraction_ok=ok,
            approved=approved,
            attempts=attempts,
            plan=plan,
            usage=usage,
        )

    def _review(self, task: CouncilTask, code: str, report: ExampleReport | None) -> _Reviewed:
        """Run the verifier, tolerating a model that can't produce a valid structured verdict."""
        try:
            res = verifier_agent.run_sync(
                _verifier_user(task, code, report),
                model=self.verifier_model,
                model_settings=self.model_settings,
            )
        except Exception:
            # A small model may fail to emit a valid Verdict even after retries. Don't crash the
            # solve: treat the candidate as unverified (the examples already passed).
            return _Reviewed(value=None, usage=Usage())
        return _Reviewed(value=res.output, usage=_usage(res))


class _Reviewed:
    """Internal: a verifier outcome plus its token cost (value is None if review failed)."""

    __slots__ = ("value", "usage")

    def __init__(self, *, value: Verdict | None, usage: Usage) -> None:
        self.value = value
        self.usage = usage


def build_council(config: CouncilConfig) -> Council:
    """Build a council whose roles call an OpenAI-compatible endpoint (standalone use).

    The bench integration injects its own models instead of going through this.
    """
    if not config.coder_model:
        raise ValueError("CouncilConfig.coder_model is required")

    def mk(handle: str) -> OpenAIChatModel:
        provider = OpenAIProvider(base_url=config.base_url, api_key=config.api_key or "not-needed")
        return OpenAIChatModel(handle, provider=provider)

    return Council(
        planner_model=mk(config.planner_model or config.coder_model),
        coder_model=mk(config.coder_model),
        verifier_model=mk(config.verifier_model or config.coder_model),
        max_retries=config.max_retries,
        model_settings=ModelSettings(temperature=config.temperature, max_tokens=config.max_tokens),
        code_runner=LocalCodeRunner(
            timeout_s=config.example_timeout_s, mem_limit_mb=config.example_mem_limit_mb
        ),
    )


def _usage(result) -> Usage:
    u = result.usage
    return Usage(
        requests=u.requests or 0,
        prompt_tokens=u.input_tokens or 0,
        completion_tokens=u.output_tokens or 0,
    )


def _planner_user(task: CouncilTask) -> str:
    parts = [task.prompt.strip(), "", f"Plan the implementation of `{task.entrypoint}`."]
    if task.signature:
        parts.append(f"Signature: {task.signature}")
    return "\n".join(parts)


def _coder_user(task: CouncilTask, plan: str, feedback: str | None) -> str:
    parts = [
        task.prompt.strip(),
        "",
        f"Define a top-level Python entrypoint named `{task.entrypoint}`.",
    ]
    if task.signature:
        parts.append(f"Expected signature/usage: {task.signature}")
    parts += ["", "Implementation plan:", plan.strip()]
    if feedback:
        parts += [
            "",
            "A previous attempt was rejected. Fix this and return the full corrected solution:",
            feedback.strip(),
        ]
    return "\n".join(parts)


def _verifier_user(task: CouncilTask, code: str, report: ExampleReport | None) -> str:
    parts = [task.prompt.strip(), "", "Candidate solution:", "```python", code.strip(), "```"]
    if report is None:
        parts += [
            "",
            "No worked examples were available to run; review by reasoning about the code against "
            "the task and its edge cases.",
        ]
    else:
        parts += [
            "",
            f"The candidate passed all {report.n} worked example(s) from the task. Review for any "
            "remaining bug or unhandled edge case.",
        ]
    return "\n".join(parts)


def _example_feedback(report: ExampleReport) -> str:
    lines = ["The solution failed these worked examples from the task:"]
    for f in report.failures[:5]:
        if f.error:
            lines.append(f"- `{f.expr}` raised {f.error}")
        else:
            lines.append(f"- `{f.expr}` was not satisfied")
    return "\n".join(lines)
