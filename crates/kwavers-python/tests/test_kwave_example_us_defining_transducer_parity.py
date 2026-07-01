#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `us_defining_transducer`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_kwave
from parity_test_utils import assert_decodable_nonblank_png, report_metric_value


skip_kwave = os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1"
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow k-wave-python tests"


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "us_defining_transducer_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("us_defining_transducer_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_trace_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert np.isfinite(metrics["peak_ratio"])
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["max_abs_diff"])


def _assert_report_contract(module, text: str):
    thresholds = module.TRACE_THRESHOLDS

    assert "parity_status: PASS" in text
    for sensor_idx in range(1, 4):
        section = f"sensor_{sensor_idx}: PASS"
        assert section in text
        assert report_metric_value(text, "pearson_r", section) >= thresholds["pearson_r"]
        rms_ratio = report_metric_value(text, "rms_ratio", section)
        assert thresholds["rms_ratio_min"] <= rms_ratio
        assert rms_ratio <= thresholds["rms_ratio_max"]
        assert np.isfinite(report_metric_value(text, "peak_ratio", section))
        assert np.isfinite(report_metric_value(text, "rmse", section))
        assert np.isfinite(report_metric_value(text, "max_abs_diff", section))


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_us_defining_transducer_artifacts_match_thresholds():
    module = _load_example_module()

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityUsDefiningTransducer:
    """Metrics-based parity coverage for the defining transducer example."""

    def test_us_defining_transducer_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kwave = result["kwave"]
        pykwavers = result["pykwavers"]

        kw_pressure = np.asarray(kwave["pressure"], dtype=np.float64)
        py_pressure = np.asarray(pykwavers["pressure"], dtype=np.float64)
        assert kw_pressure.shape == py_pressure.shape
        assert kw_pressure.shape == (3, 452)
        assert int(kwave["time_steps"]) == 452
        assert int(pykwavers["time_steps"]) == 452
        assert np.max(np.abs(kw_pressure)) > 0.0
        assert np.max(np.abs(py_pressure)) > 0.0

        for idx in range(kw_pressure.shape[0]):
            metrics = module.compute_trace_metrics(kw_pressure[idx], py_pressure[idx])
            _assert_trace_contract(metrics, module.TRACE_THRESHOLDS)

        report_lines = module.build_report_lines(kwave, pykwavers)
        _assert_report_contract(module, "\n".join(report_lines))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
