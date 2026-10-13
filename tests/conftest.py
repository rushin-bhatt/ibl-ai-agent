from __future__ import annotations

import importlib.util

import pytest

# The evals suite needs the `evals` extra (deepeval, provider SDKs). The default
# CI installs only `dev`, so skip collecting it there instead of erroring.
collect_ignore = [] if importlib.util.find_spec("deepeval") else ["evals"]

@pytest.fixture(autouse=True)
def _online_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove offline mode by default during testing"""
    monkeypatch.delenv("IBL_AGENT_DATA_OFFLINE", raising=False)