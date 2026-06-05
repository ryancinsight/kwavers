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

        assert summary["pearson_r"] > 0.999999
        assert abs(summary["rms_ratio"] - 1.0) < 1e-4
        assert summary["psnr_db"] > 90.0

        assert reference_metrics["kwave"]["pearson_r"] > 0.79
        assert reference_metrics["pykwavers"]["pearson_r"] > 0.79
        assert 0.64 <= reference_metrics["kwave"]["rms_ratio"] <= 0.70
        assert 0.64 <= reference_metrics["pykwavers"]["rms_ratio"] <= 0.70
        assert reference_metrics["kwave"]["psnr_db"] > 23.0
        assert reference_metrics["pykwavers"]["psnr_db"] > 23.0

        for row, metrics in trace_metrics.items():
            assert np.isfinite(metrics["pearson_r"]), f"sensor row {row} correlation is not finite"
            assert np.isfinite(metrics["peak_ratio"]), f"sensor row {row} peak ratio is not finite"
