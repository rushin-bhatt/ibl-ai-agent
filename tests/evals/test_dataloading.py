"""Data-loading evals: every question in ``questions/`` against every model in the grid.

Models are reached through ``actors/`` (Claude, Lightning, Mistral); deepeval owns
test cases, metrics and pass/fail. Live calls only run with ``RUN_LIVE_EVALS=1``:

    RUN_LIVE_EVALS=1 UV_CACHE_DIR=.uv-cache uv run --no-sync pytest \
        tests/evals/test_dataloading.py -n 3
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

from ibl_ai_agent.data_locations import DataLocationError, resolve_dataset_dir
from tests.evals.actors import Usage, build_actor
from tests.evals.metrics import assert_test, build_metrics
from tests.evals.skill_router import get_skill_files
from tests.evals.summarize_results import record_usage

_DIR = Path(__file__).parent
REPO_ROOT = _DIR.parents[1]


def load_questions(task: str) -> list[dict]:
    """Return every question of one ``task`` type from ``questions/*.json``."""
    questions = []
    for path in sorted(_DIR.joinpath("questions").glob("*.json")):
        questions += [q for q in json.loads(path.read_text()) if q.get("task") == task]
    return questions


def prompt_for(question: dict) -> str:
    """Build the prompt: reference material, the question, how to answer.

    Each question names ``packets`` (the Required Load Packets of AGENTS.md).
    Without them a model must guess repo conventions, such as which region
    mapping or what counts as a good unit, so the eval would measure guesswork
    rather than data loading. The answer format is fixed so that a failure
    reflects the data work, not a guess at what shape we wanted.

    Parameters
    ----------
    question : dict
        A question entry with ``question`` and optional ``packets``.

    Returns
    -------
    str
    """
    files = get_skill_files(*question.get("packets", []))
    context = "\n\n".join(f"--- {path.name} ---\n{path.read_text()}" for path in files)
    preamble = f"Use the following reference materials.\n\n{context}\n\n" if context else ""
    return (
        f"{preamble}Question: {question['question']}\n\n"
        "Compute the answer from the local IBL datasets by writing and running "
        "Python. Report every number you are asked for explicitly in your final message."
    )


QUESTIONS = load_questions("codegen")


@pytest.mark.parametrize("question", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_question(model_cfg, question):
    """One question against one model, graded by the question's own metrics."""
    try:
        resolve_dataset_dir(question["requires_data"])
    except DataLocationError:
        pytest.skip(f"dataset not available: {question['requires_data']}")

    actor = build_actor(model_cfg, cwd=REPO_ROOT)
    try:
        result = actor.run(prompt_for(question))
    except Exception as error:
        # A failed call is a result too: record it so the grid shows "error"
        # rather than "not run", then fail the test.
        record_usage(actor.name, question["id"], Usage(stop_reason=type(error).__name__), False)
        raise

    passed = False
    try:
        test_case = LLMTestCase(input=question["question"], actual_output=result.reply)
        assert_test(test_case, build_metrics(question["checks"]))
        passed = True
    finally:
        record_usage(actor.name, question["id"], result.usage, passed)
