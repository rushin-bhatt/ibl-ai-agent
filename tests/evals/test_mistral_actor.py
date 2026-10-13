"""Unit tests for the Mistral actor, with its SDK client mocked."""

import pytest

from tests.evals.actors.base import Usage
from tests.evals.actors.mistral import MistralActor
from tests.evals.fakes import chat_response, run_with_client, tool_call


def run(actor, *responses):
    return run_with_client(actor, "chat.complete", *responses)


def test_missing_usage_block_yields_zeroed_usage():
    result = run(MistralActor("m", api_key="k"), chat_response("hi", usage=None))[0]
    assert result.usage == Usage()


def test_api_errors_propagate_rather_than_scoring_as_a_wrong_answer():
    with pytest.raises(RuntimeError):
        run(MistralActor("m", api_key="k"), RuntimeError("401 unauthorized"))


def test_api_key_is_read_from_environment_when_not_passed(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    assert MistralActor("ministral-8b-latest").api_key == "from-env"


def test_tool_result_carries_the_tool_name():
    actor = MistralActor("m", api_key="k", cwd="/repo")
    _, create = run(actor, chat_response(None, [tool_call("print(1)")]), chat_response("1"))
    assert create.call_args_list[1].kwargs["messages"][2]["name"] == "run_python"
