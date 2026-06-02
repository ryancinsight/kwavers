#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `us_bmode_linear_transducer`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


METRICS_PATH = Path(__file__).resolve().parents[1] / "examples" / "output" / "us_bmode_linear_transducer_metrics.txt"


def _parse_metric_line(line: str) -> float:
    return float(line.split("=", 1)[1].strip())


@pytest.mark.timeout(30)
class TestKWaveExampleParityUsbmodeLinearTransducer:
    """Metrics-based parity coverage for the linear-transducer B-mode example."""

    def test_us_bmode_linear_transducer_quick_metrics(self):
        text = METRICS_PATH.read_text(encoding="cp1252")
        lines = text.splitlines()

        raw_idx = next(i for i, line in enumerate(lines) if line.startswith("Raw scan_lines"))
        raw_pearson = _parse_metric_line(lines[raw_idx + 1])
        raw_rms_ratio = _parse_metric_line(lines[raw_idx + 2])
        raw_psnr_db = _parse_metric_line(lines[raw_idx + 3])

        assert "parity_status: PASS" in text
        assert "Status      = PASS" in text
        assert raw_pearson > 0.97
        assert 0.90 <= raw_rms_ratio <= 1.10
        assert raw_psnr_db >= 28.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
