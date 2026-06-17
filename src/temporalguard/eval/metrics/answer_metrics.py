"""Leaderboard track: run EnterpriseRAG-Bench's own answer evaluation.

We do NOT re-implement their metrics (correctness / completeness / document-recall /
invalid-extra-documents). We run their script unmodified from the bench repo so the
numbers stay leaderboard-comparable, then read the results JSON back.

Their script (`python -m src.scripts.answer_evaluation.metrics_based_eval --answers-file ...`)
must run with cwd = the bench repo root, needs the bench's deps installed, and an LLM key
for its judges. This is gated behind `run=True` so the orchestrator can skip it (and the
LLM cost) until you explicitly want the leaderboard numbers.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_BENCH_ROOT = "EnterpriseRAG-Bench"


def run_bench_answer_eval(
    bench_answers_path: str,
    bench_root: str = DEFAULT_BENCH_ROOT,
    out_results: Optional[str] = None,
    timeout_s: int = 3600,
) -> Dict[str, Any]:
    """Run the bench's metrics_based_eval on our answers; return parsed results.

    Returns {"ran": False, "reason": ...} if the bench repo/script is unavailable,
    so the orchestrator degrades gracefully instead of crashing.
    """
    root = Path(bench_root)
    script = root / "src" / "scripts" / "answer_evaluation" / "metrics_based_eval.py"
    if not script.exists():
        return {"ran": False, "reason": f"bench script not found at {script}"}

    answers = Path(bench_answers_path).resolve()
    # Their script writes answer_evaluation/results.json by default.
    results_path = Path(out_results).resolve() if out_results else (root / "answer_evaluation" / "results.json").resolve()

    cmd = [
        "python", "-m", "src.scripts.answer_evaluation.metrics_based_eval",
        "--answers-file", str(answers),
    ]
    if out_results:
        cmd += ["--results-file", str(results_path)]

    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"ran": False, "reason": f"invocation failed: {e}"}

    if proc.returncode != 0:
        return {"ran": False, "reason": "nonzero exit", "stderr": proc.stderr[-2000:]}

    if results_path.exists():
        data = json.loads(results_path.read_text())
        return {"ran": True, "results": data, "results_path": str(results_path)}
    return {"ran": False, "reason": "no results file produced", "stdout": proc.stdout[-2000:]}
