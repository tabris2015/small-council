"""Adapter that lets slm-coding-bench drive the council as a roster solver.

This is the only module that imports slm-coding-bench, so the council core stays harness-agnostic.
It implements the bench's ``Solver`` ABC and routes every council model call through the bench's
``deployment.chat()`` (via a thin Pydantic-AI ``Model``), so the council's serving + token metrics
are recorded by the bench exactly like the built-in solvers — keeping the A/B apples-to-apples.

Loaded by the bench via an ``import_path`` solver spec, e.g. in a run config::

    solvers:
      - name: small_council
        import_path: "small_council.integrations.slm_bench:CouncilSolver"
        params: {planner_model: ..., coder_model: ..., verifier_model: ..., max_retries: 2}
"""

from __future__ import annotations

import time

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage
from slm_coding_bench.deployments.base import DeploymentAdapter
from slm_coding_bench.models import Candidate, GenMetrics, Task
from slm_coding_bench.solvers.base import Solver, SolverContext

from small_council.models import CouncilTask
from small_council.orchestrator import Council
from small_council.tool_agent import run_tool_agent_native, run_tool_agent_prompted


class _DeploymentChatModel(Model):
    """A Pydantic-AI model backed by the bench's ``deployment.chat()`` for one role's model id.

    Generation settings (temperature/max_tokens/seed) come from the bench's SolverContext, and each
    call's GenMetrics is appended to a shared list so the solver can report the per-task pipeline
    cost. Because the call goes through ``deployment.chat()``, the bench's MetricsAccumulator also
    records it for the serving-metrics table.
    """

    def __init__(
        self,
        deployment: DeploymentAdapter,
        model_id: str,
        ctx: SolverContext,
        calls: list[GenMetrics],
    ) -> None:
        super().__init__()
        self._deployment = deployment
        self._model_id = model_id
        self._ctx = ctx
        self._calls = calls

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def system(self) -> str:
        return "openai"

    async def request(self, messages, model_settings, model_request_parameters) -> ModelResponse:
        text, m = self._deployment.chat(
            model=self._model_id,
            messages=_to_chat_messages(messages),
            temperature=self._ctx.temperature,
            max_tokens=self._ctx.max_tokens,
            seed=self._ctx.seed,
        )
        self._calls.append(m)
        return ModelResponse(
            parts=[TextPart(content=text)],
            usage=RequestUsage(
                input_tokens=m.prompt_tokens or 0, output_tokens=m.completion_tokens or 0
            ),
            model_name=self._model_id,
        )


def _to_chat_messages(messages: list[ModelMessage]) -> list[dict]:
    """Flatten Pydantic-AI's message objects into OpenAI-style {role, content} dicts."""
    out: list[dict] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            if msg.instructions:  # agent instructions + PromptedOutput JSON-format guidance
                out.append({"role": "system", "content": msg.instructions})
            for part in msg.parts:
                pk = getattr(part, "part_kind", "")
                if pk == "system-prompt":
                    out.append({"role": "system", "content": _as_text(part.content)})
                elif pk == "user-prompt":
                    out.append({"role": "user", "content": _as_text(part.content)})
                elif pk == "retry-prompt":
                    out.append({"role": "user", "content": _retry_text(part)})
        elif isinstance(msg, ModelResponse):
            text = "".join(p.content for p in msg.parts if getattr(p, "part_kind", "") == "text")
            if text:
                out.append({"role": "assistant", "content": text})
    return out


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(
            c if isinstance(c, str) else getattr(c, "content", str(c)) for c in content
        )
    return str(content)


def _retry_text(part) -> str:
    render = getattr(part, "model_response", None)
    if callable(render):
        try:
            return render()
        except Exception:
            pass
    return _as_text(getattr(part, "content", ""))


class CouncilSolver(Solver):
    """slm-coding-bench solver that delegates to the small-council orchestrator."""

    name = "small_council"
    per_model = False

    def __init__(
        self,
        *,
        planner_model: str | None = None,
        coder_model: str | None = None,
        verifier_model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.verifier_model = verifier_model
        self.max_retries = max_retries

    def roster_label(self, deployment_models: list[str]) -> str:
        def short(m: str | None) -> str:
            if not m:
                return "?"
            return m.rsplit("/", 1)[-1].replace("-Instruct", "").replace("-4bit", "")

        return (
            f"council[{short(self.planner_model)}>"
            f"{short(self.coder_model)}>{short(self.verifier_model)}]"
        )

    def solve(
        self, task: Task, *, model: str, deployment: DeploymentAdapter, ctx: SolverContext
    ) -> Candidate:
        planner_id = self.planner_model or model
        coder_id = self.coder_model or model
        verifier_id = self.verifier_model or coder_id
        calls: list[GenMetrics] = []

        def mk(model_id: str) -> _DeploymentChatModel:
            return _DeploymentChatModel(deployment, model_id, ctx, calls)

        council = Council(
            planner_model=mk(planner_id),
            coder_model=mk(coder_id),
            verifier_model=mk(verifier_id),
            max_retries=self.max_retries,
        )
        result = council.solve(
            CouncilTask(
                prompt=task.prompt,
                entrypoint=task.manifest.entrypoint,
                signature=task.manifest.signature,
            )
        )
        return Candidate(
            code=result.code,
            raw_output=result.raw_output,
            extraction_ok=result.extraction_ok,
            sample_index=ctx.sample_index,
            finish_reason=None,
            gen_metrics=_combine(calls),
        )


def _combine(calls: list[GenMetrics]) -> GenMetrics | None:
    """Sum the pipeline's per-call metrics into one per-task cost record (mirrors the bench)."""
    if not calls:
        return None
    prompt = sum(c.prompt_tokens or 0 for c in calls)
    completion = sum(c.completion_tokens or 0 for c in calls)
    total_ms = sum(c.total_ms or 0.0 for c in calls)
    ttft = next((c.ttft_ms for c in calls if c.ttft_ms is not None), None)
    tps = completion / (total_ms / 1000.0) if completion and total_ms > 0 else None
    return GenMetrics(
        prompt_tokens=prompt or None,
        completion_tokens=completion or None,
        ttft_ms=ttft,
        total_ms=total_ms or None,
        tok_per_s=tps,
    )


class ToolAgentSolver(Solver):
    """A single tool-calling coder (model-driven ``run_python`` self-debug loop).

    Native tool-calling needs the model endpoint to accept OpenAI ``tools`` and return tool_calls,
    which the bench's ``deployment.chat()`` does not carry — so this points Pydantic-AI straight at
    ``base_url`` (an ``mlx_lm.server`` with the model's tool-call template). ``mode`` selects
    structured native tools vs the prompted (unstructured) baseline for the ablation.
    """

    name = "tool_agent"
    per_model = False

    def __init__(
        self,
        *,
        coder_model: str | None = None,
        base_url: str | None = None,
        api_key: str = "not-needed",
        mode: str = "native",
        max_calls: int = 6,
    ) -> None:
        self.coder_model = coder_model
        self.base_url = base_url
        self.api_key = api_key or "not-needed"
        self.mode = mode
        self.max_calls = max_calls

    def roster_label(self, deployment_models: list[str]) -> str:
        m = self.coder_model or (deployment_models[0] if deployment_models else "?")
        short = m.rsplit("/", 1)[-1].replace("-Instruct", "").replace("-4bit", "")
        return f"tool[{short}:{self.mode}]"

    def solve(
        self, task: Task, *, model: str, deployment: DeploymentAdapter, ctx: SolverContext
    ) -> Candidate:
        model_id = self.coder_model or model
        base_url = self.base_url or getattr(deployment, "base_url", None)
        pai_model = OpenAIChatModel(
            model_id, provider=OpenAIProvider(base_url=base_url, api_key=self.api_key)
        )
        ctask = CouncilTask(
            prompt=task.prompt,
            entrypoint=task.manifest.entrypoint,
            signature=task.manifest.signature,
        )
        runner = run_tool_agent_native if self.mode == "native" else run_tool_agent_prompted
        started = time.time()
        result, _stats = runner(
            ctask,
            model=pai_model,
            max_calls=self.max_calls,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
        )
        wall_ms = (time.time() - started) * 1000.0
        u = result.usage
        tps = (
            (u.completion_tokens / (wall_ms / 1000.0)) if u.completion_tokens and wall_ms else None
        )
        gm = GenMetrics(
            prompt_tokens=u.prompt_tokens or None,
            completion_tokens=u.completion_tokens or None,
            ttft_ms=None,
            total_ms=wall_ms,
            tok_per_s=tps,
        )
        return Candidate(
            code=result.code,
            raw_output=result.raw_output,
            extraction_ok=result.extraction_ok,
            sample_index=ctx.sample_index,
            finish_reason=None,
            gen_metrics=gm,
        )
