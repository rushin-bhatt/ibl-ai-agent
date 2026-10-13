"""Skill-selection evals: does a model pick the skills a question needs?

The model reads AGENTS.md and the one-line description of every repo skill, then
names the skills it would load. Graded with deepeval's ToolCorrectnessMetric
against the question's ``expected_skills``. No data or tools are needed.
"""

from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase, ToolCall

from tests.evals.actors import ActorJudge, build_actor
from tests.evals.metrics import assert_test, build_metrics
from tests.evals.summarize_results import record_usage
from tests.evals.test_dataloading import REPO_ROOT, load_questions


def skill_descriptions() -> str:
    """Return one ``- <skill>: <description>`` line per skill in ``skills/``.

    Only repo skills are listed, never ``~/.claude/skills``: the eval must not
    depend on what happens to be installed on the machine running it.
    """
    lines = []
    for skill_md in sorted(REPO_ROOT.joinpath("skills").glob("*/SKILL.md")):
        for line in skill_md.read_text(encoding="utf-8").splitlines():
            if line.startswith("description:"):
                lines.append(f"- {skill_md.parent.name}: {line.removeprefix('description:')}")
                break
    return "\n".join(lines)


def prompt_for(question: dict) -> str:
    """Build the prompt: AGENTS.md, the skill list, the question, the answer format."""
    return (
        f"{REPO_ROOT.joinpath('AGENTS.md').read_text(encoding='utf-8')}\n\n"
        f"Available skills:\n{skill_descriptions()}\n\n"
        f"Question: {question['question']}\n\n"
        "List only the skill names needed to answer this question, one per line. "
        "No explanations."
    )


QUESTIONS = load_questions("skill_selection")


@pytest.mark.parametrize("question", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_skill_selection(model_cfg, question):
    """One question against one model; picking skills is reading, so no tools."""
    actor = build_actor({**model_cfg, "tools": False}, cwd=REPO_ROOT)
    result = actor.run(prompt_for(question))
    selected = [line.strip().lstrip("- ") for line in result.reply.splitlines() if line.strip()]
    expected = [
        skill for check in question["checks"] for skill in check.get("expected_skills", [])
    ]
    test_case = LLMTestCase(
        input=question["question"],
        actual_output="",
        tools_called=[ToolCall(name=skill) for skill in selected],
        expected_tools=[ToolCall(name=skill) for skill in expected],
    )
    passed = False
    try:
        # ToolCorrectnessMetric never calls its model, but deepeval requires one;
        # the actor itself avoids a second set of credentials.
        assert_test(test_case, build_metrics(question["checks"], evaluator=ActorJudge(actor)))
        passed = True
    finally:
        record_usage(actor.name, question["id"], result.usage, passed)
