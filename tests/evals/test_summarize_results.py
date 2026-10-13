"""Unit tests for recording runs and summarizing them. No API calls."""

import pandas as pd

from tests.evals import summarize_results as results
from tests.evals.actors.base import Usage


def test_runs_append_under_one_header_and_summarize_per_model(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results, "USAGE_CSV", tmp_path.joinpath("usage.csv"))
    results.record_usage("haiku", "q1", Usage(input_tokens=10, cost_usd=0.5), True)
    results.record_usage("haiku", "q2", Usage(stop_reason="max_turns"), False)
    results.record_usage("mistral", "q1", Usage(stop_reason="RuntimeError"), False)

    runs = pd.read_csv(tmp_path.joinpath("usage.csv"))
    assert list(runs.columns) == results.COLUMNS
    summary = results.summarize(runs)
    assert summary.loc["haiku", "q2"] == "fail (max_turns)"
    assert summary.loc["mistral", "q2"] == "not run"  # never reached, not failed
    assert summary.loc["haiku", "pass %"] == 50
    assert (summary.loc["haiku", "cost"], summary.loc["mistral", "cost"]) == (
        "$0.50", results.NO_COST)
