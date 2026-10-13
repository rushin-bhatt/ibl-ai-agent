"""Fake chat responses and tool calls for the actor unit tests."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def chat_response(content, tool_calls=None, finish_reason="stop", usage=(11, 7)):
    """A fake OpenAI/Mistral-shaped chat response; ``usage=None`` omits the block."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    tokens = usage and SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=tokens)


def tool_call(code, call_id="call_1"):
    """A tool call asking to run ``code``, with JSON arguments as the APIs send them."""
    function = SimpleNamespace(name="run_python", arguments=json.dumps({"code": code}))
    return SimpleNamespace(id=call_id, function=function)


def run_with_client(actor, create_path, *responses):
    """Run ``actor`` against a mocked SDK client, run_python patched to print "1".

    Parameters
    ----------
    actor : ChatCompletionActor
    create_path : str
        Attribute path of the client's completion call, e.g. "chat.complete".
    *responses
        Returned (or raised, for exceptions) by successive calls.

    Returns
    -------
    tuple
        ``(RunResult, the completion-call mock)``.
    """
    client = MagicMock()
    create = client
    for attr in create_path.split("."):
        create = getattr(create, attr)
    create.side_effect = list(responses)
    with patch.object(actor, "_client", return_value=client), patch(
        "tests.evals.actors.tools.run_python", return_value="1"
    ):
        return actor.run("how many sessions?"), create
