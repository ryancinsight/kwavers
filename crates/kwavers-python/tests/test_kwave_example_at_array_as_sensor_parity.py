#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `at_array_as_sensor`.
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
    module_path = root / "examples" / "at_array_as_sensor_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("at_array_as_sensor_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_metric_contract(metric, thresholds):
    assert metric["pearson_r"] >= thresholds["pearson_r"]
    if "psnr_db" in thresholds:
        assert metric["psnr_db"] > thresholds["psnr_db"]


def _assert_trace_summary_contract(summary, thresholds):
    assert summary["pearson_r_min"] >= thresholds["pearson_r_min"]
    assert summary["pearson_r_mean"] >= thresholds["pearson_r_mean"]
    assert summary["rms_ratio_min"] >= thresholds["rms_ratio_min"]
    assert summary["rms_ratio_max"] <= thresholds["rms_ratio_max"]
    assert summary["peak_ratio_min"] >= thresholds["peak_ratio_min"]
    assert summary["peak_ratio_max"] <= thresholds["peak_ratio_max"]
    assert summary["rmse_max"] < thresholds["rmse_max"]


def _assert_report_section_contract(text: str, section: str, thresholds):
    assert report_metric_value(text, "pearson_r", section) >= thresholds["pearson_r"]
    if "psnr_db" in thresholds:
        assert report_metric_value(text, "psnr_db", section) > thresholds["psnr_db"]


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS

    assert "parity_status: PASS" in text
    _assert_report_section_contract(
        text,
        "sensor mask parity:",
        thresholds["mask_metrics"],
    )
    _assert_report_section_contract(
        text,
        "weighted mask parity:",
        thresholds["weighted_mask_metrics"],
    )
    _assert_report_section_contract(
        text,
        "raw detector matrix parity:",
        thresholds["raw_matrix_metrics"],
    )
    _assert_report_section_contract(
        text,
        "combined arc-trace parity:",
        thresholds["combined_matrix_metrics"],
    )

    trace = thresholds["trace_summary"]
    assert report_metric_value(text, "combined trace pearson_r_min") >= trace["pearson_r_min"]
    assert report_metric_value(text, "combined trace pearson_r_mean") >= trace["pearson_r_mean"]
    assert report_metric_value(text, "combined trace rms_ratio_min") >= trace["rms_ratio_min"]
    assert report_metric_value(text, "combined trace rms_ratio_max") <= trace["rms_ratio_max"]
    assert report_metric_value(text, "combined trace peak_ratio_min") >= trace["peak_ratio_min"]
    assert report_metric_value(text, "combined trace peak_ratio_max") <= trace["peak_ratio_max"]
    assert report_metric_value(text, "combined trace rmse_max") < trace["rmse_max"]


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_at_array_as_sensor_artifacts_match_thresholds():
    module = _load_example_module()
    text = module.METRICS_PATH.read_text(encoding="utf-8")

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityAtArrayAsSensor:
    """Metrics-based parity coverage for the array-as-sensor example."""

    def test_at_array_as_sensor_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kwave = result["kwave"]
        pykwavers = result["pykwavers"]
        layout = result["layout"]
        mask_metrics = result["mask_metrics"]
        weighted_mask_metrics = result["weighted_mask_metrics"]
        raw_matrix_metrics = result["raw_matrix_metrics"]
        combined_matrix_metrics = result["combined_matrix_metrics"]
        trace_summary = result["trace_summary"]
        trace_metrics = result["trace_metrics"]

        sensor_mask_kw = np.asarray(layout["sensor_mask_kw"], dtype=bool)
        sensor_mask_pk = np.asarray(layout["sensor_mask_pk"], dtype=bool)
        sensor_weighted_mask_kw = np.asarray(layout["sensor_weighted_mask_kw"], dtype=np.float64)
        sensor_weighted_mask_pk = np.asarray(layout["sensor_weighted_mask_pk"], dtype=np.float64)
        kw_pressure = np.asarray(kwave["pressure"], dtype=np.float64)
        py_pressure = np.asarray(pykwavers["pressure"], dtype=np.float64)
        kw_combined = np.asarray(kwave["combined"], dtype=np.float64)
        py_combined = np.asarray(pykwavers["combined"], dtype=np.float64)

        assert sensor_mask_kw.shape == sensor_mask_pk.shape
        assert sensor_weighted_mask_kw.shape == sensor_weighted_mask_pk.shape
        assert np.array_equal(sensor_mask_kw, sensor_mask_pk)
        assert np.allclose(sensor_weighted_mask_kw, sensor_weighted_mask_pk, rtol=1e-9, atol=1e-12)

        assert kw_pressure.shape == py_pressure.shape
        assert kw_combined.shape == py_combined.shape
        assert kw_pressure.ndim == 2
        assert kw_combined.ndim == 2
        assert kw_pressure.shape[0] > 0
        assert kw_pressure.shape[1] > 0
        assert kw_combined.shape[0] == 20
        assert np.max(np.abs(kw_pressure)) > 0.0
        assert np.max(np.abs(py_pressure)) > 0.0
        assert np.max(np.abs(kw_combined)) > 0.0
        assert np.max(np.abs(py_combined)) > 0.0

        thresholds = module.PARITY_THRESHOLDS
        _assert_metric_contract(mask_metrics, thresholds["mask_metrics"])
        _assert_metric_contract(weighted_mask_metrics, thresholds["weighted_mask_metrics"])
        _assert_metric_contract(raw_matrix_metrics, thresholds["raw_matrix_metrics"])
        _assert_metric_contract(
            combined_matrix_metrics,
            thresholds["combined_matrix_metrics"],
        )
        _assert_trace_summary_contract(trace_summary, thresholds["trace_summary"])

        for metrics in trace_metrics.values():
            assert metrics["pearson_r"] >= thresholds["trace_summary"]["pearson_r_min"]
            assert metrics["rms_ratio"] >= thresholds["trace_summary"]["rms_ratio_min"]
            assert metrics["rms_ratio"] <= thresholds["trace_summary"]["rms_ratio_max"]
            assert metrics["peak_ratio"] >= thresholds["trace_summary"]["peak_ratio_min"]
            assert metrics["peak_ratio"] <= thresholds["trace_summary"]["peak_ratio_max"]
            assert metrics["rmse"] < thresholds["trace_summary"]["rmse_max"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
