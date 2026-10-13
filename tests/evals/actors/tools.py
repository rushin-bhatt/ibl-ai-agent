"""The one tool chat-only actors get: run Python in the repo environment.

Claude's SDK brings its own tools. Lightning and Mistral are plain chat endpoints,
so to answer a data-loading question they need some way to execute code. This
module provides the minimum: a ``run_python`` function, its JSON schema for
function calling, and the parser for the arguments a model sends to it.

Model-written code runs on the machine running the evals, with no sandbox. That
is inherent to the task ("load this dataset and compute a number"), but it means
evals belong on a disposable machine, such as a CI runner, and never against an
untrusted question set.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

_SCRATCH_HOME = str(Path(tempfile.gettempdir()).joinpath("ibl-eval-home"))

TIMEOUT_S = 180
MAX_OUTPUT_CHARS = 10_000

# Function-calling schema in the shape both OpenAI-compatible and Mistral APIs
# expect. The description says how to reach the datasets, because a chat model
# cannot explore the repo the way an agent with file tools can.
RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python in the ibl-ai-agent repo environment and return stdout "
            "and stderr. pandas is available. Resolve dataset directories with: "
            "from ibl_ai_agent.data_locations import resolve_dataset_dir; "
            "root = resolve_dataset_dir('bwm_ephys'). Tables are parquet files "
            "under root, e.g. metadata/units.parquet. Print what you want to see."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source."}},
            "required": ["code"],
        },
    },
}


def run_python(code: str, cwd) -> str:
    """Run ``code`` with the repo's interpreter and return its output as text.

    A timeout or crash is returned as text, not raised: the model should see its
    own error and get a chance to fix it, which is what the turn limit is for.

    Parameters
    ----------
    code : str
        Python source to execute.
    cwd : str or Path
        Repo root; the interpreter is ``<cwd>/.venv/bin/python``.

    Returns
    -------
    str
        stdout then stderr, truncated to ``MAX_OUTPUT_CHARS`` so one careless
        print of a whole dataframe cannot flood the context.
    """
    try:
        completed = subprocess.run(
            [".venv/bin/python", "-c", code],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            # A minimal environment: model code gets no API keys, and HOME points
            # outside the repo so ONE and matplotlib caches land in temp rather
            # than the working tree (one run fetched 39 MB into the repo root).
            env={
                "UV_CACHE_DIR": ".uv-cache",
                "PATH": "/usr/bin:/bin",
                "HOME": _SCRATCH_HOME,
                "MPLCONFIGDIR": _SCRATCH_HOME,
            },
        )
    except subprocess.TimeoutExpired:
        return f"TimeoutError: code ran longer than {TIMEOUT_S} s"
    output = (completed.stdout + completed.stderr).strip() or "(no output)"
    if len(output) > MAX_OUTPUT_CHARS:
        output = f"{output[:MAX_OUTPUT_CHARS]}\n...[truncated]"
    return output


def parse_code_argument(call) -> str:
    """Return the ``code`` argument of a ``run_python`` tool call.

    Parameters
    ----------
    call : object
        A tool call with ``.function.arguments``, a JSON string or a decoded dict.

    Returns
    -------
    str
        The code. Malformed JSON becomes code that fails loudly, so the model
        sees the error instead of the loop crashing.
    """
    arguments = call.function.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return "raise SystemExit('tool call arguments were not valid JSON')"
    return arguments.get("code", "")
