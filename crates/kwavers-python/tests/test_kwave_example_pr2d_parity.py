#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for the 2D FFT line-sensor example.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_KWAVE, requires_kwave
from parity_test_utils import assert_decodable_nonblank_png, report_metric_value


skip_kwave = os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1"
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow k-wave-python tests"


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "compare_pr_2D_FFT_line_sensor.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("compare_pr_2D_FFT_line_sensor", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_metrics_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert metrics["psnr_db"] >= thresholds["psnr_db"]


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS
    reconstruction_section = "FFT line reconstruction (kwave vs pykwavers):"
    reference_section = "Reconstruction vs ground-truth p0:"

    assert "parity_status: PASS" in text
    assert (
        report_metric_value(text, "pearson_r", reconstruction_section)
        >= thresholds["reconstruction"]["pearson_r"]
    )
    rms_ratio = report_metric_value(text, "rms_ratio", reconstruction_section)
    assert thresholds["reconstruction"]["rms_ratio_min"] <= rms_ratio
    assert rms_ratio <= thresholds["reconstruction"]["rms_ratio_max"]
    assert (
        report_metric_value(text, "psnr_db", reconstruction_section)
        >= thresholds["reconstruction"]["psnr_db"]
    )

    labels = {
        "kwave": "kwave     ",
        "pykwavers": "pykwavers ",
    }
    for prefix, label in labels.items():
        assert (
            report_metric_value(text, f"{label}pearson_r", reference_section)
            >= thresholds["reference"]["pearson_r"]
        )
        reference_rms = report_metric_value(text, f"{label}rms_ratio", reference_section)
        assert thresholds["reference"]["rms_ratio_min"] <= reference_rms
        assert reference_rms <= thresholds["reference"]["rms_ratio_max"]
        assert (
            report_metric_value(text, f"{label}psnr_db", reference_section)
            >= thresholds["reference"]["psnr_db"]
        )


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_pr_2d_fft_line_sensor_artifacts_match_thresholds():
    module = _load_example_module()

    assert module.METRICS_PATH.exists()
    assert module.RECON_FIGURE_PATH.exists()
    assert module.PRESSURE_FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.RECON_FIGURE_PATH)
    assert_decodable_nonblank_png(module.PRESSURE_FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParity2D:
    """Metrics-based parity coverage for the vendored 2D FFT line-sensor example."""

    def test_pr_2d_fft_line_sensor_metrics(self):
        """Exact line-sensor initial-pressure example matches on reconstruction metrics."""
        module = _load_example_module()
        result = module.run_comparison()

        kw_pressure = np.asarray(result["kwave"]["pressure"], dtype=np.float64)
        py_pressure = np.asarray(result["pykwavers"]["pressure"], dtype=np.float64)
        kw_reconstruction = np.asarray(result["kwave"]["reconstruction"], dtype=np.float64)
        py_reconstruction = np.asarray(result["pykwavers"]["reconstruction"], dtype=np.float64)
        summary = result["summary"]
        reference_metrics = result["reference_metrics"]
        trace_metrics = result["trace_metrics"]

        assert kw_pressure.shape == py_pressure.shape, (
            f"shape mismatch: {kw_pressure.shape} != {py_pressure.shape}"
        )
        assert kw_reconstruction.shape == py_reconstruction.shape
        assert np.all(np.isfinite(kw_pressure))
        assert np.all(np.isfinite(py_pressure))
        assert np.all(np.isfinite(kw_reconstruction))
        assert np.all(np.isfinite(py_reconstruction))
        assert np.max(np.abs(kw_pressure)) > 0.0
        assert np.max(np.abs(py_pressure)) > 0.0
        assert np.max(np.abs(kw_reconstruction)) > 0.0
        assert np.max(np.abs(py_reconstruction)) > 0.0

        thresholds = module.PARITY_THRESHOLDS
        _assert_metrics_contract(summary, thresholds["reconstruction"])
        _assert_metrics_contract(reference_metrics["kwave"], thresholds["reference"])
        _assert_metrics_contract(reference_metrics["pykwavers"], thresholds["reference"])

        for row, metrics in trace_metrics.items():
            assert np.isfinite(metrics["pearson_r"]), f"sensor row {row} correlation is not finite"
            assert np.isfinite(metrics["peak_ratio"]), f"sensor row {row} peak ratio is not finite"
