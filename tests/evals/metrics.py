"""Metrics for the evals suite, built from the ``checks`` entries of question files.

Also holds ``assert_test``, the one wrapper around deepeval's assertion that every
eval module uses, so its console table goes to the log instead of the terminal.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re

from deepeval import assert_test as _deepeval_assert_test
from deepeval.metrics import BaseMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase

log = logging.getLogger(__name__)


class NeuronCountMetric(BaseMetric):
    """Pass when every expected count appears in the answer as a whole number.

    Parameters
    ----------
    expected : dict[str, list[int]]
        Label -> counts that must all appear, e.g. ``{"CB": [7273, 53660]}``.
        Labels are for humans; only the numbers are matched.
    """

    name = "NeuronCountMetric"

    def __init__(self, expected: dict[str, list[int]]) -> None:
        self.threshold = 1.0
        self.score = 0.0
        self.expected = expected

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Score 1.0 if all expected counts are in ``actual_output``, else 0.0.

        Thousands separators are removed first. Numbers are matched whole: a
        plain substring test passes "5" against the "5" inside "2657", a silent
        false pass on short expected values.
        """
        clean = re.sub(r"(\d),(\d)", r"\1\2", test_case.actual_output)
        missing = [
            str(n)
            for counts in self.expected.values()
            for n in counts
            if not re.search(rf"(?<!\d){n}(?!\d)", clean)
        ]
        self.score = 0.0 if missing else 1.0
        preview = test_case.actual_output.replace("\n", " | ")[:200]
        self.reason = f"Missing counts: {missing} — output: {preview}" if missing else "All present"
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)


def build_metrics(checks: list[dict], evaluator=None) -> list[BaseMetric]:
    """Instantiate the metrics a question declares.

    Parameters
    ----------
    checks : list[dict]
        The question's ``checks``, each with a ``metric`` name and its settings.
    evaluator : deepeval.models.DeepEvalBaseLLM, optional
        Judge model for metrics that require one. Without it, deepeval falls back
        to its default model, which needs OPENAI_API_KEY.

    Returns
    -------
    list[BaseMetric]
        One metric per check, in order.

    Raises
    ------
    ValueError
        For a metric name this module does not know.
    """
    metrics: list[BaseMetric] = []
    for check in checks:
        if check["metric"] == "NeuronCountMetric":
            metrics.append(NeuronCountMetric(expected=check["expected"]))
        elif check["metric"] == "ToolCorrectnessMetric":
            metrics.append(
                ToolCorrectnessMetric(threshold=check.get("threshold", 1.0), model=evaluator)
            )
        else:
            raise ValueError(f"Unknown metric: {check['metric']!r}")
    return metrics


def assert_test(test_case: LLMTestCase, metrics: list[BaseMetric]) -> None:
    """Run deepeval's ``assert_test``, logging its report at DEBUG.

    Raises
    ------
    AssertionError
        When any metric fails, exactly as deepeval does.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _deepeval_assert_test(test_case=test_case, metrics=metrics)
    finally:
        if report := buf.getvalue():
            log.debug("deepeval report:\n%s", report)
