"""Eval actors: one interface over three model providers.

Build one from config with `build_actor`:

    build_actor({"provider": "claude", "model": "haiku"}, cwd=repo_root)
    build_actor({"provider": "lightning", "model": "lightning-ai/gemma-4-31B-it"})
    build_actor({"provider": "mistral", "model": "ministral-8b-latest"})

A model grid is then a list of such dicts, and adding a provider means adding
one Actor subclass and one entry in PROVIDERS.
"""

from .base import Actor, CodeExecutingActor, RunResult, Usage
from .chat import ChatCompletionActor
from .claude_sdk import ClaudeSDKActor
from .deepeval_adapter import ActorJudge
from .lightning import LightningActor
from .mistral import MistralActor

PROVIDERS = {
    "claude": ClaudeSDKActor,
    "lightning": LightningActor,
    "mistral": MistralActor,
}


def build_actor(config, cwd=None):
    """Build an Actor from a grid entry.

    Parameters
    ----------
    config : dict
        ``{"provider": str, "model": str}`` plus optional ``name``,
        ``max_output_tokens``, ``max_turns`` and ``tools`` (default True).
        ``tools: false`` builds a chat actor without run_python: the no-data
        baseline.
    cwd : str or Path, optional
        Repo root. Required by code-executing actors; passed to chat actors only
        when they get tools, which is what lets them run code.

    Returns
    -------
    Actor

    Raises
    ------
    KeyError
        For an unknown provider, so a typo in a grid fails instead of silently
        skipping a model.
    """
    config = dict(config)
    provider = config.pop("provider")
    use_tools = config.pop("tools", True)
    actor_class = PROVIDERS[provider]
    if issubclass(actor_class, CodeExecutingActor):
        return actor_class(cwd=cwd, **config)
    return actor_class(cwd=cwd if use_tools else None, **config)


__all__ = [
    "Actor",
    "ActorJudge",
    "ChatCompletionActor",
    "CodeExecutingActor",
    "ClaudeSDKActor",
    "LightningActor",
    "MistralActor",
    "RunResult",
    "Usage",
    "PROVIDERS",
    "build_actor",
]
