"""Unit tests for the run_python tool. Runs real subprocesses; no network or keys."""

import sys
from types import SimpleNamespace

import pytest

from tests.evals.actors import tools


@pytest.fixture
def repo(tmp_path):
    """A directory whose ``.venv/bin/python`` is the interpreter running the tests."""
    tmp_path.joinpath(".venv", "bin").mkdir(parents=True)
    tmp_path.joinpath(".venv", "bin", "python").symlink_to(sys.executable)
    return tmp_path


def test_returns_stdout_and_stderr(repo):
    output = tools.run_python("import sys; print(444); print('warn', file=sys.stderr)", repo)
    assert output == "444\nwarn"


def test_errors_come_back_as_text_for_the_model_to_fix(repo):
    assert "ZeroDivisionError" in tools.run_python("1 / 0", repo)


def test_timeout_comes_back_as_text(repo, monkeypatch):
    monkeypatch.setattr(tools, "TIMEOUT_S", 0.5)
    assert tools.run_python("import time; time.sleep(5)", repo).startswith("TimeoutError")


def test_long_output_is_truncated(repo):
    output = tools.run_python(f"print('x' * {tools.MAX_OUTPUT_CHARS * 2})", repo)
    assert output.endswith("[truncated]")
    assert len(output) < tools.MAX_OUTPUT_CHARS + 50


def test_model_code_cannot_read_api_keys(repo, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "secret")
    assert tools.run_python("import os; print(os.getenv('MISTRAL_API_KEY'))", repo) == "None"


def call_with(arguments):
    """A fake tool call carrying ``arguments``."""
    return SimpleNamespace(function=SimpleNamespace(name="run_python", arguments=arguments))


@pytest.mark.parametrize("arguments", ['{"code": "print(1)"}', {"code": "print(1)"}])
def test_code_argument_parsed_from_json_or_dict(arguments):
    assert tools.parse_code_argument(call_with(arguments)) == "print(1)"


def test_malformed_arguments_become_code_that_fails_loudly():
    assert "not valid JSON" in tools.parse_code_argument(call_with("{not json"))
