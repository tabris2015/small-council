"""A minimal, self-contained sandbox that runs candidate code against worked examples.

Writes the candidate + a stdlib-only driver + the example asserts into a fresh temp dir, runs the
driver in a subprocess with a scrubbed environment and best-effort resource limits, and reads back
a JSON value channel. No third-party deps; safe to use standalone. The bench can inject its own
hardened executor instead (both satisfy the :class:`~small_council.ports.CodeRunner` protocol).
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

from small_council.models import ExampleReport, ExampleResult

# stdlib-only; loads the candidate, evaluates each boolean expr, writes result.json as the channel.
_CHECK_DRIVER = """
import json, sys
ns = {}
try:
    with open("solution.py") as fh:
        exec(compile(fh.read(), "solution.py", "exec"), ns)
except Exception as e:
    json.dump({"loaded": False, "error": repr(e)}, open("result.json", "w"))
    sys.exit(0)
spec = json.load(open(sys.argv[1]))
results = []
for expr in spec["asserts"]:
    try:
        ok = bool(eval(expr, dict(ns)))
        results.append({"expr": expr, "ok": ok})
    except Exception as e:
        results.append({"expr": expr, "ok": False, "error": repr(e)})
json.dump({"loaded": True, "results": results}, open("result.json", "w"))
"""


def _preexec(cpu_time_s: int, mem_limit_mb: int | None):
    """Return a preexec_fn that isolates the child and applies best-effort resource limits.

    RLIMIT_AS is skipped on macOS: the system Python needs a large address space just to start, so
    capping it there tends to kill the interpreter rather than the workload. CPU time is reliable
    everywhere and is the real backstop against infinite loops alongside the wall-clock timeout.
    """

    def _set() -> None:
        os.setsid()
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_s, cpu_time_s))
        except (ValueError, OSError):
            pass
        if mem_limit_mb is not None and sys.platform != "darwin":
            nbytes = mem_limit_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
            except (ValueError, OSError):
                pass

    return _set


class LocalCodeRunner:
    """Default :class:`~small_council.ports.CodeRunner` backed by a subprocess sandbox."""

    def __init__(self, timeout_s: float = 10.0, mem_limit_mb: int | None = 1024) -> None:
        self.timeout_s = timeout_s
        self.mem_limit_mb = mem_limit_mb

    def run_examples(self, code: str, examples: list[str]) -> ExampleReport:
        if not examples:
            return ExampleReport(n=0)
        with tempfile.TemporaryDirectory(prefix="council-") as d:
            tmp = Path(d)
            (tmp / "solution.py").write_text(code)
            (tmp / "check.py").write_text(_CHECK_DRIVER)
            (tmp / "spec.json").write_text(json.dumps({"asserts": examples}))
            env = {"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": ""}
            cpu = max(1, int(self.timeout_s) + 1)
            try:
                proc = subprocess.run(
                    [sys.executable, "check.py", "spec.json"],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    preexec_fn=_preexec(cpu, self.mem_limit_mb),
                )
            except subprocess.TimeoutExpired:
                return ExampleReport(n=len(examples), timed_out=True)

            rj_path = tmp / "result.json"
            if not rj_path.exists():
                detail = (proc.stderr or "no result.json produced")[:500]
                return ExampleReport(n=len(examples), loaded=False, load_error=detail)
            rj = json.loads(rj_path.read_text())
            if not rj.get("loaded", False):
                return ExampleReport(
                    n=len(examples), loaded=False, load_error=rj.get("error", "load failed")
                )
            results = [
                ExampleResult(expr=r["expr"], ok=bool(r.get("ok")), error=r.get("error"))
                for r in rj.get("results", [])
            ]
            return ExampleReport(n=len(examples), loaded=True, results=results)
