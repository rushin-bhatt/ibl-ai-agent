"""Unit tests for the Claude actor, with the SDK's messages faked. No CLI or keys."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.evals.actors.claude_sdk import ClaudeSDKActor


def result_message(subtype="success", usage=None, cost=0.0335, num_turns=6):
    """A fake ``ResultMessage``."""
    return SimpleNamespace(subtype=subtype, usage=usage, total_cost_usd=cost, num_turns=num_turns)


def test_resolves_model_aliases_but_keeps_display_name():
    actor = ClaudeSDKActor("haiku", cwd="/repo")
    assert (actor.model, actor.name) == ("claude-haiku-4-5", "haiku")
    assert ClaudeSDKActor("claude-opus-5", cwd="/repo").model == "claude-opus-5"


def test_usage_includes_cache_tokens_and_cost():
    raw = {"input_tokens": 42, "output_tokens": 2554, "cache_read_input_tokens": 93019,
           "cache_creation_input_tokens": 5153}
    usage = ClaudeSDKActor._usage_from(result_message(usage=raw))
    assert (usage.input_tokens, usage.output_tokens) == (42, 2554)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (93019, 5153)
    assert (usage.cost_usd, usage.num_turns, usage.stop_reason) == (0.0335, 6, "answered")


def test_missing_cost_reads_as_not_reported():
    # A subscription session can report no cost: 0.0 means "not reported", not "free".
    usage = ClaudeSDKActor._usage_from(result_message(usage={}, cost=None))
    assert (usage.cost_usd, usage.input_tokens) == (0.0, 0)


class ResultError(Exception):
    """Stand-in for the SDK's ResultError, raised after the final message."""

    def __init__(self, subtype):
        super().__init__(subtype)
        self.subtype = subtype


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a fake ``claude_agent_sdk`` whose query yields a result, then raises."""
    sdk = ModuleType("claude_agent_sdk")
    sdk.AssistantMessage, sdk.TextBlock = type("A", (), {}), type("T", (), {})
    sdk.ResultMessage, sdk.ResultError = SimpleNamespace, ResultError
    sdk.ClaudeAgentOptions = lambda **kwargs: kwargs

    def install(subtype):
        async def query(prompt, options):
            yield result_message(subtype=subtype, usage={})
            raise ResultError(subtype)

        sdk.query = query
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)

    return install


def test_running_out_of_turns_is_recorded_not_raised(fake_sdk):
    # The SDK raises on max turns; that is a result, like a chat actor's max_turns.
    fake_sdk("error_max_turns")
    assert ClaudeSDKActor("haiku", cwd="/repo").run("q").usage.stop_reason == "max_turns"


def test_other_errors_raise(fake_sdk):
    fake_sdk("error_during_execution")
    with pytest.raises(ResultError):
        ClaudeSDKActor("haiku", cwd="/repo").run("q")
