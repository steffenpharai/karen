"""E2E tests for vitals pipeline with synthetic frames."""

import numpy as np
import pytest
from vision.vitals import VitalsAnalyzer, VitalsResult


@pytest.mark.e2e
def test_vitals_analyzer_synthetic_frame():
    """VitalsAnalyzer produces a result from a synthetic frame."""
    analyzer = VitalsAnalyzer(fps=30)
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(frame)
    assert isinstance(result, VitalsResult)
    assert result.timestamp > 0


@pytest.mark.e2e
def test_vitals_multiple_frames():
    """VitalsAnalyzer accumulates data across frames."""
    analyzer = VitalsAnalyzer(fps=30)
    for _ in range(10):
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = analyzer.analyze(frame)
    assert isinstance(result, VitalsResult)


@pytest.mark.e2e
def test_vitals_last_result():
    """last_result returns the most recent analysis."""
    analyzer = VitalsAnalyzer(fps=30)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = analyzer.analyze(frame)
    assert analyzer.last_result is result
    assert analyzer.last_result.timestamp == result.timestamp
