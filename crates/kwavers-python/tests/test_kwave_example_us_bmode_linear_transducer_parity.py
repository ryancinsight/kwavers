#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `us_bmode_linear_transducer`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from parity_test_utils import assert_decodable_nonblank_png


OUTPUT = Path(__file__).resolve().parents[1] / "examples" / "output"
METRICS_PATH = OUTPUT / "us_bmode_linear_transducer_metrics.txt"
COMPARISON_PNG = OUTPUT / "us_bmode_linear_transducer_compare.png"
KWAVE_PNG = OUTPUT / "us_bmode_linear_transducer_kwave.png"
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _raw_scan_line_block(text: str) -> str:
    match = re.search(
        r"(?ms)^Raw scan_lines.*?(?=^GPU scan-line profile|\Z)",
        text,
    )
    assert match is not None
    return match.group(0)


def _metric_value(block: str, label: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(label)}\s*=\s*({_FLOAT})", block)
    assert match is not None, label
    return float(match.group(1))


def _target_values(block: str) -> dict[str, float]:
    match = re.search(
        rf"Targets\s*=\s*r>=({_FLOAT}),\s*"
        rf"({_FLOAT})<=RMS<=({_FLOAT}),\s*"
        rf"PSNR>=({_FLOAT})\s*dB",
        block,
    )
    assert match is not None
    return {
        "pearson_r": float(match.group(1)),
        "rms_ratio_min": float(match.group(2)),
        "rms_ratio_max": float(match.group(3)),
        "psnr_db": float(match.group(4)),
    }


@pytest.mark.timeout(30)
class TestKWaveExampleParityUsbmodeLinearTransducer:
    """Metrics-based parity coverage for the linear-transducer B-mode example."""

    def test_us_bmode_linear_transducer_quick_metrics(self):
        text = METRICS_PATH.read_text(encoding="cp1252")
        raw_block = _raw_scan_line_block(text)

        raw_pearson = _metric_value(raw_block, "Pearson r")
        raw_rms_ratio = _metric_value(raw_block, "RMS ratio")
        raw_psnr_db = _metric_value(raw_block, "PSNR [dB]")
        targets = _target_values(raw_block)

        assert "parity_status: PASS" in text
        assert "Status      = PASS" in raw_block
        assert raw_pearson >= targets["pearson_r"]
        assert targets["rms_ratio_min"] <= raw_rms_ratio
        assert raw_rms_ratio <= targets["rms_ratio_max"]
        assert raw_psnr_db >= targets["psnr_db"]
        assert_decodable_nonblank_png(COMPARISON_PNG)
        assert_decodable_nonblank_png(KWAVE_PNG)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
