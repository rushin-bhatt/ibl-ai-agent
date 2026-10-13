"""Unit tests for the Lightning actor, with its OpenAI client mocked."""

import pytest

from tests.evals.actors.lightning import LightningActor
from tests.evals.fakes import chat_response, run_with_client, tool_call


def run(actor, *responses):
    return run_with_client(actor, "chat.completions.create", *responses)


def test_sends_prompt_and_parses_reply():
    result, create = run(LightningActor("lightning-ai/gemma-4-31B-it", api_key="k"),
                         chat_response("the answer"))
    assert create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "how many sessions?"}]
    assert (result.reply, result.usage.input_tokens, result.usage.output_tokens) == (
        "the answer", 11, 7)


@pytest.mark.parametrize(
    "model, kwarg",
    [("openai/gpt-5-nano", "max_completion_tokens"), ("o3-mini", "max_completion_tokens"),
     ("lightning-ai/gemma-4-31B-it", "max_tokens")],
)
def test_picks_the_token_limit_argument_each_model_accepts(model, kwarg):
    # Newer OpenAI-lineage models reject max_tokens; older ones reject the other.
    actor = LightningActor(model, api_key="k")
    assert run(actor, chat_response("ok"))[1].call_args.kwargs[kwarg] == actor.max_output_tokens


def test_reasoning_model_returning_no_content_gives_empty_reply():
    # Seen with Qwen3.8-27B and gpt-5-nano: reasoning spends the budget and
    # content comes back None. That is an empty answer, not a crash.
    assert run(LightningActor("m", api_key="k"), chat_response(None))[0].reply == ""


def test_encodes_tool_turns_in_openai_shape():
    actor = LightningActor("openai/gpt-5-nano", api_key="k", cwd="/repo")
    _, create = run(actor, chat_response(None, [tool_call("print(1)")]), chat_response("1"))
    history = create.call_args_list[1].kwargs["messages"]
    assert history[1]["tool_calls"][0]["type"] == "function"  # OpenAI requires type
    assert history[2] == {"role": "tool", "tool_call_id": "call_1", "content": "1"}
