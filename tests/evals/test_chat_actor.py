"""Unit tests for the chat tool loop in ``actors/chat.py``, with a fake provider."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.evals.actors.base import Usage
from tests.evals.actors.chat import ChatCompletionActor


def tool_call(code, call_id="call_1"):
    """A tool call asking to run ``code``, with JSON arguments as the APIs send them."""
    function = SimpleNamespace(name="run_python", arguments=json.dumps({"code": code}))
    return SimpleNamespace(id=call_id, function=function)


class FakeChatActor(ChatCompletionActor):
    """Replays scripted ``(content, tool_calls, usage)`` turns and records requests."""

    def __init__(self, turns, **kwargs):
        super().__init__("fake-model", api_key="k", **kwargs)
        self.turns = iter(turns)
        self.requests = []

    def _client(self):
        return None

    def _complete(self, client, messages, tool_schemas):
        self.requests.append({"messages": list(messages), "tools": tool_schemas})
        return next(self.turns)

    def _assistant_message(self, content, tool_calls):
        return {"role": "assistant", "content": content}

    def _tool_message(self, call, output):
        return {"role": "tool", "content": output}


def run(actor, output="444"):
    """Run ``actor`` with run_python patched to return ``output``; return (result, mock)."""
    with patch("tests.evals.actors.tools.run_python", return_value=output) as run_python:
        return actor.run("how many sessions?"), run_python


def test_runs_requested_code_and_feeds_output_back():
    actor = FakeChatActor(
        [("", [tool_call("print(444)")], Usage()), ("There are 444.", [], Usage())], cwd="/repo"
    )
    result, run_python = run(actor)
    run_python.assert_called_once_with("print(444)", "/repo")
    assert actor.requests[1]["messages"][-1] == {"role": "tool", "content": "444"}
    assert result.reply == "There are 444."
    assert result.usage.stop_reason == "answered"


def test_sums_tokens_and_counts_turns_across_the_loop():
    actor = FakeChatActor(
        [
            ("", [tool_call("print(1)")], Usage(input_tokens=100, output_tokens=20)),
            ("done", [], Usage(input_tokens=150, output_tokens=30)),
        ],
        cwd="/repo",
    )
    usage = run(actor)[0].usage
    assert (usage.input_tokens, usage.output_tokens, usage.num_turns) == (250, 50, 2)


def test_no_tools_offered_without_cwd():
    actor = FakeChatActor([("I cannot run code.", [], Usage())])
    run(actor)
    assert actor.requests[0]["tools"] is None
    assert actor.executes_code is False


def test_stops_at_max_turns():
    # A model that only ever calls tools must not loop forever.
    turns = [("", [tool_call("print(1)")], Usage())] * 5
    actor = FakeChatActor(turns, cwd="/repo", max_turns=3)
    result = run(actor)[0]
    assert len(actor.requests) == 3
    assert (result.usage.num_turns, result.usage.stop_reason) == (3, "max_turns")


@pytest.mark.parametrize(
    "content, provider_reason, expected",
    [("the answer", "stop", "answered"), ("", "malformed_function_call", "malformed_function_call")],
)
def test_stop_reason_tells_an_answer_from_nothing_usable(content, provider_reason, expected):
    # Gemini 2.5 Flash rejects its own tool call on long prompts: no content, no
    # call. That must not read as an empty answer.
    actor = FakeChatActor([(content, [], Usage(stop_reason=provider_reason))])
    assert run(actor)[0].usage.stop_reason == expected
