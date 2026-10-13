"""Pytest configuration for the evals suite.

Every test that takes a ``model_cfg`` fixture is parametrized over the model grid,
so pytest expands it into a model x question matrix. Such tests call a provider
API, cost tokens and need keys, so they only run with ``RUN_LIVE_EVALS=1``.

Grid sources, first match wins:

1. ``EVAL_MODELS_FILE``, a path to a grid file (used by the nightly workflow)
2. ``tests/evals/models.json``, a user-local grid (gitignored)
3. ``tests/evals/models.default.json``, one model per provider (committed)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from dotenv import load_dotenv

_DIR = Path(__file__).parent

load_dotenv(_DIR.joinpath(".env.local"), override=False)

# (model_id, question_id, outcome) for every test that ran in this process.
_results: list[tuple[str, str, str]] = []


def load_models() -> list[dict]:
    """Return the model grid.

    Returns
    -------
    list[dict]
        One config per model, each with ``provider`` and ``model`` keys and
        optional actor arguments. Empty when no grid file exists.
    """
    if os.getenv("EVAL_MODELS_FILE"):
        return json.loads(Path(os.environ["EVAL_MODELS_FILE"]).read_text())
    for name in ("models.json", "models.default.json"):
        if _DIR.joinpath(name).exists():
            return json.loads(_DIR.joinpath(name).read_text())
    return []


def model_id(cfg: dict) -> str:
    """Return ``provider/model``, the id a model has in test ids and tables."""
    return "/".join(p for p in (cfg.get("provider", ""), cfg.get("model", "")) if p)


def parse_nodeid(nodeid: str) -> tuple[str, str] | None:
    """Split a parametrized test id into ``(model_id, question_id)``.

    Pytest joins parameter ids with "-", and both halves contain hyphens of their
    own (``mistral/ministral-8b-latest-bwm-neuron-count-by-region``), so the
    split point cannot be found by pattern alone. Model ids are known from the
    grid, so match the longest known prefix.

    Parameters
    ----------
    nodeid : str
        A pytest node id such as ``test_x.py::test_question[<model>-<question>]``.

    Returns
    -------
    tuple[str, str] or None
        ``(model_id, question_id)``, or None for a test that is not parametrized.
    """
    match = re.search(r"\[(.+)\]$", nodeid)
    if not match:
        return None
    params = match.group(1)
    for mid in sorted((model_id(cfg) for cfg in load_models()), key=len, reverse=True):
        if mid and params.startswith(f"{mid}-"):
            return mid, params[len(mid) + 1 :]
    # Unknown grid: split at the last hyphen rather than lose the row.
    model, _, question = params.rpartition("-")
    return (model, question) if model else None


def pytest_generate_tests(metafunc) -> None:
    """Parametrize every test that takes ``model_cfg`` over the grid."""
    if "model_cfg" in metafunc.fixturenames:
        models = load_models()
        metafunc.parametrize("model_cfg", models, ids=[model_id(m) for m in models])


def pytest_collection_modifyitems(config, items) -> None:
    """Skip tests that call a live model unless ``RUN_LIVE_EVALS=1``.

    Opting in keeps ``just check`` offline and free; the nightly workflow sets it.
    """
    if os.getenv("RUN_LIVE_EVALS") == "1":
        return
    skip = pytest.mark.skip(reason="live eval: set RUN_LIVE_EVALS=1 to call model APIs")
    for item in items:
        if "model_cfg" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)


def pytest_runtest_logreport(report) -> None:
    """Record the outcome of each grid test for the terminal summary."""
    if report.when != "call":
        return
    parsed = parse_nodeid(report.nodeid)
    if parsed is None:
        return
    outcome = "PASS" if report.passed else "SKIP" if report.skipped else "FAIL"
    _results.append((*parsed, outcome))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Print a model x question grid of pass/fail marks after the run."""
    if not _results:
        return
    models = list(dict.fromkeys(m for m, _, _ in _results))
    questions = list(dict.fromkeys(q for _, q, _ in _results))
    lookup = {(m, q): o for m, q, o in _results}
    symbols = {"PASS": "✓", "FAIL": "✗", "SKIP": "-"}
    col_w = max(len(q) for q in questions) + 2
    model_w = max(len(m) for m in models) + 2

    terminalreporter.write_sep("=", "eval results")
    header = f"{'model':<{model_w}}" + "".join(f"{q:^{col_w}}" for q in questions)
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))
    for model in models:
        marks = "".join(f"{symbols[lookup.get((model, q), 'SKIP')]:^{col_w}}" for q in questions)
        terminalreporter.write_line(f"{model:<{model_w}}{marks}")
