"""Unit tests for the eval metrics. No API calls.

    UV_CACHE_DIR=.uv-cache uv run --no-sync pytest tests/evals/test_metrics.py
"""

import pytest
from deepeval.test_case import LLMTestCase

from tests.evals.metrics import NeuronCountMetric, build_metrics


def measure(expected, output):
    """Return whether ``output`` passes a NeuronCountMetric for ``expected``."""
    metric = NeuronCountMetric(expected=expected)
    metric.measure(LLMTestCase(input="q", actual_output=output))
    return metric.is_successful()


def test_counts_present_in_any_format_pass():
    expected = {"CB": [7273, 53660]}
    assert measure(expected, "CB has 7273 good units of 53660 total")
    assert measure(expected, "CB: 7,273 good / 53,660 total")  # thousands separators


def test_missing_count_fails():
    assert not measure({"CB": [7273, 53660]}, "CB has 7273 good units")


def test_short_count_is_not_matched_inside_a_longer_number():
    # Regression: a substring test passes "5" against the "5" in "2657".
    assert not measure({"sessions": [5]}, "left 2657, right 437, both 13")
    assert measure({"sessions": [5]}, "both hemispheres: 5 sessions")


def test_count_is_not_matched_as_a_prefix_of_a_longer_number():
    assert not measure({"units": [201]}, "there are 2018 units")


def test_build_metrics_rejects_an_unknown_metric():
    # A typo in a question file must fail loudly, not silently grade nothing.
    with pytest.raises(ValueError):
        build_metrics([{"metric": "NeuronCuontMetric"}])
