"""Record each eval run, and turn the log into a summary table.

Tests call ``record_usage`` as they go; this script renders the result:

    UV_CACHE_DIR=.uv-cache uv run --no-sync python tests/evals/summarize_results.py

It writes ``summary.md`` next to ``usage.csv``: one row per model, one column per
question, then tokens and cost. Both live in ``EVAL_RESULTS_DIR``, by default
``tests/evals/results/`` (gitignored).
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent
RESULTS_DIR = Path(os.getenv("EVAL_RESULTS_DIR", _DIR.joinpath("results")))
USAGE_CSV = RESULTS_DIR.joinpath("usage.csv")
SUMMARY_MD = RESULTS_DIR.joinpath("summary.md")

# timestamp and commit place each run on a time axis and a code version, which is
# what a results database or dashboard groups by.
COLUMNS = ["timestamp", "commit", "model", "question", "passed", "input_tokens",
           "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd",
           "num_turns", "stop_reason"]

# Cost is only reported by the Claude CLI; Mistral and Lightning return none.
NO_COST = "not reported"


def current_commit() -> str:
    """Return the short commit under test: ``GITHUB_SHA`` on CI, else git's HEAD."""
    sha = os.getenv("GITHUB_SHA") or subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_DIR, capture_output=True, text=True
    ).stdout.strip()
    return sha[:7] or "unknown"


def record_usage(model_name, question_id, usage, passed) -> None:
    """Append one run to ``usage.csv``.

    Appending per run, rather than at the end, means an interrupted grid keeps
    its completed rows.

    Parameters
    ----------
    model_name : str
        The actor's name.
    question_id : str
        The question's ``id``.
    usage : tests.evals.actors.Usage
        Tokens, cost, turns and stop reason of the run.
    passed : bool
        Whether every metric passed.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive create, so that parallel workers (-n) write the header once.
    try:
        with USAGE_CSV.open("x", newline="") as handle:
            csv.writer(handle).writerow(COLUMNS)
    except FileExistsError:
        pass
    with USAGE_CSV.open("a", newline="") as handle:
        csv.writer(handle).writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"), current_commit(),
            model_name, question_id, passed, usage.input_tokens, usage.output_tokens,
            usage.cache_read_tokens, usage.cache_write_tokens, round(usage.cost_usd, 6),
            usage.num_turns, usage.stop_reason,
        ])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Pivot runs into one row per model.

    Parameters
    ----------
    runs : pd.DataFrame
        ``usage.csv`` as read, one row per run, columns ``COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Index model; one column per question holding "pass" or
        "fail (<stop_reason>)", then token totals, cost and pass %. Input tokens
        include cache reads, which dominate for agents that re-read context.
    """
    runs = runs.assign(
        input_total=runs.input_tokens + runs.cache_read_tokens,
        outcome=runs.passed.map({True: "pass"}).fillna("fail (" + runs.stop_reason + ")"),
    )
    by_question = runs.pivot_table(
        index="model", columns="question", values="outcome", aggfunc="first"
    ).fillna("not run")  # never reached is not the same as failed
    totals = runs.groupby("model").agg(
        passed=("passed", "sum"),
        n_runs=("passed", "size"),
        input_tokens=("input_total", "sum"),
        output_tokens=("output_tokens", "sum"),
        cost_usd=("cost_usd", "sum"),
    )
    summary = by_question.join(totals)
    summary["pass %"] = (100 * summary.passed / summary.n_runs).round(0).astype(int)
    summary["cost"] = summary.cost_usd.map(lambda c: f"${c:.2f}" if c else NO_COST)
    return summary.sort_values(["pass %", "model"], ascending=[False, True])


def to_markdown(summary: pd.DataFrame, runs: pd.DataFrame) -> str:
    """Render ``summarize`` output as markdown, under a one-line header of totals."""
    questions = sorted(runs.question.unique())
    table = summary[questions + ["pass %", "input_tokens", "output_tokens", "cost"]].rename(
        columns={"input_tokens": "input tokens (incl. cache)", "output_tokens": "output tokens"}
    )
    passed, total = int(runs.passed.sum()), len(runs)
    return (
        f"# Eval grid results\n\n"
        f"{len(summary)} models x {len(questions)} questions, {total} runs, "
        f"{passed} passed ({100 * passed // total}%).\n\n"
        f"{table.to_markdown()}\n"
    )


def main():
    if not USAGE_CSV.exists():
        sys.exit(f"no results at {USAGE_CSV} — run the grid first")
    runs = pd.read_csv(USAGE_CSV)
    report = to_markdown(summarize(runs), runs)
    SUMMARY_MD.write_text(report)
    print(report)
    print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
