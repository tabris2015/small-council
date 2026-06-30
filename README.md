# small-council

A heterogeneous, verifier-checked **multi-agent coder built on small LLMs** (1–7B). A *council* of
specialized small models — a **planner**, a **coder**, and a **verifier** — collaborate on a coding
task instead of relying on one model in one shot.

This is the experimental subject of an open research question: *does a heterogeneous,
supervisor-anchored, verifier-checked multi-agent design beat a single small agent at coding, at
this scale?* It is evaluated by the companion benchmark harness
[`slm-coding-bench`](https://github.com/tabris2015/slm-coding-bench).

## Design

```
Planner   (e.g. Qwen2.5-Coder-7B)  -> a short implementation plan (no code)
Coder     (e.g. Qwen2.5-Coder-3B)  -> implement the plan as code
Verifier  (e.g. Qwen3-1.7B)        -> run the candidate against worked examples + review;
                                      APPROVE, or send concrete feedback back to the coder
                                      (bounded retries)
```

- **Heterogeneous + moderately sparse**, not a flat swarm — to suppress the small-model
  error-compounding / tool-reliability cliff.
- **Orchestrator-driven (v0):** a fixed pipeline calls each role; the models do not autonomously
  call tools yet. Real LLM tool-calling + constrained decoding is the planned **v1** milestone.
- **Built on [Pydantic-AI](https://ai.pydantic.dev/):** typed agent roles, an OpenAI-compatible
  model layer, and structured outputs (the verifier returns a typed `Verdict`).
- **Harness-agnostic core:** the package depends on nothing benchmark-specific. A thin integration
  shim (added later) lets `slm-coding-bench` drive it as a solver.

## Status

Early scaffold. The stable core is in place — typed I/O contracts (`models.py`), config, the
`CodeRunner` port, a self-contained sandbox runner, and a code extractor. The agent roles,
orchestrator, and CLI land next.

## Evaluation

small-council is measured by the companion harness
[`slm-coding-bench`](https://github.com/tabris2015/slm-coding-bench). The adapter in
`small_council/integrations/slm_bench.py` implements the bench's `Solver` interface and routes every
model call through the bench's deployment, so the council's pass@1 / parity / perf and serving
metrics are recorded just like the built-in solvers. From a `slm-coding-bench` checkout with
small-council importable in its environment (e.g. `uv pip install -e ../small-council`):

```bash
uv run slm-bench run -c configs/m4-agent-system.yaml --run-id council
uv run slm-bench report baseline naive-multi council   # merged A/B comparison table
```

(The adapter is the only module that imports `slm-coding-bench`; the rest of the package has no
benchmark dependency.)

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).
