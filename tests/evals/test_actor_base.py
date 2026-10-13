"""Unit tests for the actor base types. No API calls."""

import pytest

from tests.evals.actors.base import Actor, CodeExecutingActor, Usage


def test_usage_adds_tokens_and_cost_but_keeps_turns_and_stop_reason():
    total = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5, num_turns=3,
                  stop_reason="max_turns")
    total = total + Usage(input_tokens=10, output_tokens=20, cache_read_tokens=5, cost_usd=0.25)
    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens) == (11, 22, 5)
    assert (total.cost_usd, total.num_turns, total.stop_reason) == (0.75, 3, "max_turns")


def test_actor_name_defaults_to_model():
    class Echo(CodeExecutingActor):
        def run(self, prompt):
            return prompt

    assert Echo("m", cwd="/repo").name == "m"
    assert Echo("m", cwd="/repo", name="label").name == "label"
    assert Echo.executes_code is True


def test_actor_without_run_cannot_be_built():
    with pytest.raises(TypeError):
        Actor("m")
