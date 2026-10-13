"""Chat-endpoint actors: a model behind a chat API, given run_python and a tool loop.

Providers differ only in how they encode requests and tool calls; subclasses
implement those four methods and share the loop.
"""

from __future__ import annotations

from abc import abstractmethod

from . import tools
from .base import Actor, RunResult, Usage


class ChatCompletionActor(Actor):
    """A chat endpoint, optionally given ``run_python`` and a tool loop.

    With ``cwd`` set, the actor loops: call the model, run any code it asks for,
    feed the output back, until it answers or hits ``max_turns``. That makes a
    data-loading question answerable by a chat endpoint, and comparable with an
    agent. Without ``cwd`` no tool is offered: the no-data baseline.

    ``max_output_tokens`` defaults high because reasoning models spend tokens
    before answering; on a small budget they return empty content.

    Subclasses implement the four provider-specific methods below.
    """

    def __init__(self, model, api_key, name=None, max_output_tokens=4096, cwd=None,
                 max_turns=12):
        super().__init__(model, name)
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.cwd = cwd
        self.max_turns = max_turns

    @property
    def executes_code(self):
        """True when this actor was given the run_python tool."""
        return self.cwd is not None

    @abstractmethod
    def _client(self):
        """Return the provider SDK client. Separate so tests can patch it."""

    @abstractmethod
    def _complete(self, client, messages, tool_schemas):
        """Make one API call; return ``(content, tool_calls, usage)``.

        ``content`` is "" when the model only called tools; ``tool_calls`` is a
        list of objects with ``.id`` and ``.function.name/.arguments``; ``usage``
        has ``stop_reason`` set from the provider's finish reason.
        """

    @abstractmethod
    def _assistant_message(self, content, tool_calls):
        """Re-encode the model's own turn for the next request's history."""

    @abstractmethod
    def _tool_message(self, call, output):
        """Encode one tool result for the next request's history."""

    def run(self, prompt):
        """Answer one question, running the tool loop when tools are enabled."""
        client = self._client()
        tool_schemas = [tools.RUN_PYTHON_TOOL] if self.executes_code else None
        messages = [{"role": "user", "content": prompt}]
        total, reply = Usage(), ""

        for turn in range(1, self.max_turns + 1):
            content, tool_calls, usage = self._complete(client, messages, tool_schemas)
            total = total + usage
            total.num_turns = turn
            reply = content or reply
            if not tool_calls:
                # No text and no tool call: the provider returned nothing usable
                # (e.g. Gemini's "malformed_function_call"). Keep its reason.
                total.stop_reason = "answered" if content else usage.stop_reason or "empty"
                break
            messages.append(self._assistant_message(content, tool_calls))
            for call in tool_calls:
                output = tools.run_python(tools.parse_code_argument(call), self.cwd)
                messages.append(self._tool_message(call, output))
        else:
            total.stop_reason = "max_turns"

        return RunResult(reply=reply, usage=total)
