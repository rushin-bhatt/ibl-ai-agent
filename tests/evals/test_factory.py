"""Unit tests for ``build_actor`` and the committed model grids. No API calls."""

import json
from pathlib import Path

import pytest

from tests.evals.actors import (
    PROVIDERS,
    ClaudeSDKActor,
    LightningActor,
    MistralActor,
    build_actor,
)


def test_builds_each_provider():
    assert isinstance(build_actor({"provider": "claude", "model": "haiku"}, cwd="/r"),
                      ClaudeSDKActor)
    assert isinstance(build_actor({"provider": "lightning", "model": "m", "api_key": "k"}),
                      LightningActor)
    assert isinstance(build_actor({"provider": "mistral", "model": "m", "api_key": "k"}),
                      MistralActor)


def test_rejects_unknown_provider():
    with pytest.raises(KeyError):
        build_actor({"provider": "nope", "model": "m"})


def test_tools_flag_controls_whether_a_chat_actor_can_run_code():
    cfg = {"provider": "mistral", "model": "m", "api_key": "k"}
    assert build_actor(cfg, cwd="/r").executes_code is True
    assert build_actor({**cfg, "tools": False}, cwd="/r").executes_code is False
    # Claude is an agent: it always runs code, whatever the flag.
    assert build_actor({"provider": "claude", "model": "haiku"}, cwd="/r").executes_code


@pytest.mark.parametrize("grid", ["models.default.json", "models.grid.json"])
def test_committed_grids_only_name_known_providers(grid):
    # A grid entry for a provider that does not exist fails every test it reaches.
    entries = json.loads(Path(__file__).parent.joinpath(grid).read_text())
    assert {entry["provider"] for entry in entries} <= set(PROVIDERS)
