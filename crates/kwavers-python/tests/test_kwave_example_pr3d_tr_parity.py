#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for the 3D time-reversal planar-sensor example.
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
    module_path = root / "examples" / "compare_pr_3D_TR_planar_sensor.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("compare_pr_3D_TR_planar_sensor", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_metrics_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    if "rms_ratio_min" in thresholds:
        assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    if "rms_ratio_max" in thresholds:
        assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    if "psnr_db" in thresholds:
        assert metrics["psnr_db"] >= thresholds["psnr_db"]


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS
    tr_section = "Time-reversal reconstruction (kwave vs pykwavers):"
    reference_section = "Reconstruction vs ground-truth p0:"

    assert "parity_status: PASS" in text
    tr_thresholds = thresholds["time_reversal"]
    assert report_metric_value(text, "pearson_r", tr_section) >= tr_thresholds["pearson_r"]
    rms_ratio = report_metric_value(text, "rms_ratio", tr_section)
    assert tr_thresholds["rms_ratio_min"] <= rms_ratio
    assert rms_ratio <= tr_thresholds["rms_ratio_max"]
    assert report_metric_value(text, "psnr_db", tr_section) >= tr_thresholds["psnr_db"]

    reference_thresholds = thresholds["reference"]
    for label in ("kwave     pearson_r", "pykwavers pearson_r"):
        assert report_metric_value(text, label, reference_section) >= reference_thresholds["pearson_r"]


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_pr_3d_time_reversal_planar_sensor_artifacts_match_thresholds():
    module = _load_example_module()

    assert module.METRICS_PATH.exists()
    assert module.TR_FIGURE_PATH.exists()
    assert module.PRESSURE_FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.TR_FIGURE_PATH)
    assert_decodable_nonblank_png(module.PRESSURE_FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParity3DTR:
    """Metrics-based parity coverage for the vendored 3D time-reversal example."""

    def test_pr_3d_time_reversal_planar_sensor_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kw_pressure = np.asarray(result["kwave"]["pressure"], dtype=np.float64)
        py_pressure = np.asarray(result["pykwavers"]["pressure"], dtype=np.float64)
        kw_tr = np.asarray(result["kwave"]["time_reversal"], dtype=np.float64)
        py_tr = np.asarray(result["pykwavers"]["time_reversal"], dtype=np.float64)
        summary = result["summary"]
        reference_metrics = result["reference_metrics"]
        trace_metrics = result["trace_metrics"]

        assert kw_pressure.shape == py_pressure.shape, (
            f"shape mismatch: {kw_pressure.shape} != {py_pressure.shape}"
        )
        assert kw_tr.shape == py_tr.shape
        assert np.all(np.isfinite(kw_pressure))
        assert np.all(np.isfinite(py_pressure))
        assert np.all(np.isfinite(kw_tr))
        assert np.all(np.isfinite(py_tr))
        assert np.max(np.abs(kw_pressure)) > 0.0
        assert np.max(np.abs(py_pressure)) > 0.0
        assert np.max(np.abs(kw_tr)) > 0.0
        assert np.max(np.abs(py_tr)) > 0.0

        thresholds = module.PARITY_THRESHOLDS
        _assert_metrics_contract(summary, thresholds["time_reversal"])
        _assert_metrics_contract(reference_metrics["kwave_time_reversal"], thresholds["reference"])
        _assert_metrics_contract(reference_metrics["pykwavers_time_reversal"], thresholds["reference"])

        for row, metrics in trace_metrics.items():
            assert np.isfinite(metrics["pearson_r"]), f"sensor row {row} correlation is not finite"
            assert np.isfinite(metrics["peak_ratio"]), f"sensor row {row} peak ratio is not finite"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
