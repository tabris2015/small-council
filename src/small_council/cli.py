"""Standalone CLI: run the council on a single coding task and print the solution.

Example:
    council solve "Implement add(a, b) returning the sum." -e add \\
        --planner mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \\
        --coder   mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \\
        --verifier mlx-community/Qwen3-1.7B-4bit \\
        --base-url http://127.0.0.1:18080/v1
"""

from __future__ import annotations

import typer

from small_council.config import CouncilConfig
from small_council.models import CouncilTask
from small_council.orchestrator import build_council

app = typer.Typer(
    help="small-council: a multi-agent coder built on small LLMs.", no_args_is_help=True
)


@app.command()
def solve(
    prompt: str = typer.Argument(..., help="the coding task / problem statement"),
    entrypoint: str = typer.Option(..., "--entrypoint", "-e", help="function/class name to define"),
    signature: str | None = typer.Option(None, "--signature", help="expected signature/usage hint"),
    planner: str = typer.Option(..., "--planner", help="planner model handle"),
    coder: str = typer.Option(..., "--coder", help="coder model handle"),
    verifier: str = typer.Option(..., "--verifier", help="verifier model handle"),
    base_url: str = typer.Option(
        "http://127.0.0.1:8080/v1", "--base-url", envvar="OPENAI_BASE_URL"
    ),
    api_key: str = typer.Option("not-needed", "--api-key", envvar="OPENAI_API_KEY"),
    max_retries: int = typer.Option(2, "--max-retries"),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
) -> None:
    """Run the council on one task; print the code to stdout and a one-line summary to stderr."""
    config = CouncilConfig(
        planner_model=planner,
        coder_model=coder,
        verifier_model=verifier,
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    council = build_council(config)
    result = council.solve(CouncilTask(prompt=prompt, entrypoint=entrypoint, signature=signature))
    typer.echo(result.code)
    typer.echo(
        f"\n# approved={result.approved} attempts={result.attempts} "
        f"extraction_ok={result.extraction_ok} "
        f"tokens(in/out)={result.usage.prompt_tokens}/{result.usage.completion_tokens}",
        err=True,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
