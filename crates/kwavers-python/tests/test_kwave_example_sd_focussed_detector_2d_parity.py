#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for the 2D focussed detector example.
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
    module_path = root / "examples" / "sd_focussed_detector_2D_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("sd_focussed_detector_2D_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_trace_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert metrics["rmse"] <= thresholds["rmse"]
    assert thresholds["peak_ratio_min"] <= metrics["peak_ratio"]
    assert metrics["peak_ratio"] <= thresholds["peak_ratio_max"]


def _assert_directivity_contract(metrics, thresholds):
    assert metrics["kwave_ratio"] > thresholds["ratio_min"]
    assert metrics["pykwavers_ratio"] > thresholds["ratio_min"]
    assert metrics["relative_error"] <= thresholds["relative_error"]


def _assert_report_contract(module, text: str):
    trace_thresholds = module.PARITY_THRESHOLDS["trace"]
    directivity_thresholds = module.PARITY_THRESHOLDS["directivity"]

    assert "parity_status: PASS" in text
    assert report_metric_value(text, "kwave_directivity_ratio") > directivity_thresholds["ratio_min"]
    assert report_metric_value(text, "pykwavers_directivity_ratio") > directivity_thresholds["ratio_min"]
    assert report_metric_value(text, "directivity_relative_error") <= directivity_thresholds["relative_error"]

    for section in ("on_axis:", "off_axis:"):
        assert report_metric_value(text, "pearson_r", section) >= trace_thresholds["pearson_r"]
        rms_ratio = report_metric_value(text, "rms_ratio", section)
        assert trace_thresholds["rms_ratio_min"] <= rms_ratio
        assert rms_ratio <= trace_thresholds["rms_ratio_max"]
        assert report_metric_value(text, "rmse", section) <= trace_thresholds["rmse"]
        peak_ratio = report_metric_value(text, "peak_ratio", section)
        assert trace_thresholds["peak_ratio_min"] <= peak_ratio
        assert peak_ratio <= trace_thresholds["peak_ratio_max"]


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_sd_focussed_detector_2d_artifacts_match_thresholds():
    module = _load_example_module()

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    assert module.DIRECTIVITY_FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)
    assert_decodable_nonblank_png(module.DIRECTIVITY_FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParitySDFocussedDetector2D:
    """Metrics-based parity coverage for the 2D focussed detector example."""

    def test_sd_focussed_detector_2d_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kw_on = np.asarray(result["kwave"]["on_axis"]["trace"], dtype=np.float64)
        kw_off = np.asarray(result["kwave"]["off_axis"]["trace"], dtype=np.float64)
        py_on = np.asarray(result["pykwavers"]["on_axis"]["trace"], dtype=np.float64)
        py_off = np.asarray(result["pykwavers"]["off_axis"]["trace"], dtype=np.float64)
        trace_metrics = result["trace_metrics"]
        directivity = result["directivity_metrics"]

        assert kw_on.shape == py_on.shape
        assert kw_off.shape == py_off.shape
        assert kw_on.ndim == 1
        assert kw_off.ndim == 1
        assert np.all(np.isfinite(kw_on))
        assert np.all(np.isfinite(py_on))
        assert np.all(np.isfinite(kw_off))
        assert np.all(np.isfinite(py_off))
        assert np.max(np.abs(kw_on)) > 0.0
        assert np.max(np.abs(py_on)) > 0.0
        assert np.max(np.abs(kw_off)) > 0.0
        assert np.max(np.abs(py_off)) > 0.0

        thresholds = module.PARITY_THRESHOLDS
        for tag in ("on_axis", "off_axis"):
            _assert_trace_contract(trace_metrics[tag], thresholds["trace"])

        _assert_directivity_contract(directivity, thresholds["directivity"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
