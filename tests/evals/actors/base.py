"""Base classes for eval actors.

An *actor* is the model under test: it receives a question and returns an answer
plus token usage. Every provider exposes the same interface, so the grid runner
never branches on which API it is talking to.

    Actor                      what every provider must offer
    |- CodeExecutingActor      an agent with its own tools (Claude)
    `- ChatCompletionActor     a chat endpoint with a tool loop (chat.py)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field



@dataclass
class Usage:
    """Token counts and cost for one run, normalised across providers.

    ``cost_usd`` is 0.0 when the provider reports none: absence of a figure, not
    a claim that the call was free. ``stop_reason`` says why the run ended:
    "answered", "max_turns", or a provider finish reason such as
    "malformed_function_call". A wrong answer and no answer are different results.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 1
    stop_reason: str = "answered"

    def __add__(self, other: Usage) -> Usage:
        """Sum token counts and cost; keep this run's turn count and stop reason."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            num_turns=self.num_turns,
            stop_reason=self.stop_reason,
        )


@dataclass
class RunResult:
    """What an actor returns: the model's reply text plus its usage."""

    reply: str
    usage: Usage = field(default_factory=Usage)


class Actor(ABC):
    """One model behind one provider.

    Parameters
    ----------
    model : str
        Provider model id.
    name : str, optional
        Label in result tables, unique within a grid. Defaults to ``model``.
    """

    executes_code = False

    def __init__(self, model: str, name: str | None = None):
        self.model = model
        self.name = name or model

    @abstractmethod
    def run(self, prompt: str) -> RunResult:
        """Answer one question.

        API errors must raise: a failed call is a real result and must not be
        scored as a wrong answer.
        """

    def __repr__(self):
        return f"{type(self).__name__}(model={self.model!r})"


class CodeExecutingActor(Actor):
    """An agent that runs code and reads files in ``cwd``, the repo root."""

    executes_code = True

    def __init__(self, model, cwd, name=None, max_turns=30):
        super().__init__(model, name)
        self.cwd = cwd
        self.max_turns = max_turns
