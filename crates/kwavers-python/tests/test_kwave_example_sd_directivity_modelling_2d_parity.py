#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for the 2-D sensor directivity model.
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
    module_path = root / "examples" / "sd_directivity_modelling_2D_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("sd_directivity_modelling_2D_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_matrix_contract(metrics, thresholds):
    assert metrics["pearson_r"] > thresholds["pearson_r"]
    assert metrics["psnr_db"] > thresholds["psnr_db"]


def _assert_trace_summary_contract(metrics, thresholds):
    assert metrics["pearson_r_min"] > thresholds["pearson_r_min"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio_mean"]
    assert metrics["rms_ratio_mean"] <= thresholds["rms_ratio_max"]


def _assert_directivity_contract(metrics, thresholds):
    assert metrics["pearson_r"] > thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert thresholds["peak_ratio_min"] <= metrics["peak_ratio"]
    assert metrics["peak_ratio"] <= thresholds["peak_ratio_max"]


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS

    assert "parity_status: PASS" in text
    assert report_metric_value(text, "matrix_pearson_r") > thresholds["matrix"]["pearson_r"]
    assert report_metric_value(text, "matrix_psnr_db") > thresholds["matrix"]["psnr_db"]
    assert report_metric_value(text, "trace_pearson_r_min") > thresholds["trace"]["pearson_r_min"]
    trace_rms = report_metric_value(text, "trace_rms_ratio_mean")
    assert thresholds["trace"]["rms_ratio_min"] <= trace_rms
    assert trace_rms <= thresholds["trace"]["rms_ratio_max"]
    assert report_metric_value(text, "directivity_pearson_r") > thresholds["directivity"]["pearson_r"]
    directivity_rms = report_metric_value(text, "directivity_rms_ratio")
    assert thresholds["directivity"]["rms_ratio_min"] <= directivity_rms
    assert directivity_rms <= thresholds["directivity"]["rms_ratio_max"]
    peak_ratio = report_metric_value(text, "directivity_peak_ratio")
    assert thresholds["directivity"]["peak_ratio_min"] <= peak_ratio
    assert peak_ratio <= thresholds["directivity"]["peak_ratio_max"]


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_sd_directivity_modelling_2d_artifacts_match_thresholds():
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
class TestKWaveExampleParitySDDirectivityModelling2D:
    """Metrics-based parity coverage for the 2-D sensor directivity example."""

    def test_sd_directivity_modelling_2d_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        aligned = result["aligned"]
        kw_traces = np.asarray(aligned["kw_traces"], dtype=np.float64)
        py_traces = np.asarray(aligned["py_traces"], dtype=np.float64)
        kw_directivity = np.asarray(aligned["kw_directivity"], dtype=np.float64)
        py_directivity = np.asarray(aligned["py_directivity"], dtype=np.float64)
        matrix_metrics = result["matrix_metrics"]
        trace_summary = result["trace_summary"]
        directivity = result["directivity_metrics"]

        assert kw_traces.shape == py_traces.shape
        assert kw_traces.ndim == 2
        assert kw_traces.shape[0] == 11
        assert kw_traces.shape[1] > 0
        assert np.all(np.isfinite(kw_traces))
        assert np.all(np.isfinite(py_traces))
        assert np.max(np.abs(kw_traces)) > 0.0
        assert np.max(np.abs(py_traces)) > 0.0

        assert kw_directivity.shape == py_directivity.shape
        assert kw_directivity.shape == (11,)
        assert np.all(np.isfinite(kw_directivity))
        assert np.all(np.isfinite(py_directivity))
        assert np.max(kw_directivity) > 0.0
        assert np.max(py_directivity) > 0.0

        thresholds = module.PARITY_THRESHOLDS
        _assert_matrix_contract(matrix_metrics, thresholds["matrix"])
        _assert_trace_summary_contract(trace_summary, thresholds["trace"])
        _assert_directivity_contract(directivity, thresholds["directivity"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
