#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for pykwavers.

This suite exercises exact example reproductions against the vendored
k-wave-python examples and asserts on metrics, not only on successful execution.
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
    module_path = root / "examples" / "compare_pr_3D_FFT_planar_sensor.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("compare_pr_3D_FFT_planar_sensor", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParity:
    """Metrics-based parity coverage for vendored k-wave-python examples."""

    def test_pr_3d_fft_planar_sensor_metrics(self):
        """Exact planar-sensor initial-pressure example matches on trace metrics."""
        module = _load_example_module()
        result = module.run_comparison()

        kw_pressure = np.asarray(result["kwave"]["pressure"], dtype=np.float64)
        py_pressure = np.asarray(result["pykwavers"]["pressure"], dtype=np.float64)
        summary = result["summary"]
        trace_metrics = result["trace_metrics"]

        assert kw_pressure.shape == py_pressure.shape, (
            f"shape mismatch: {kw_pressure.shape} != {py_pressure.shape}"
        )
        assert np.all(np.isfinite(kw_pressure))
        assert np.all(np.isfinite(py_pressure))
        assert np.max(np.abs(kw_pressure)) > 0.0
        assert np.max(np.abs(py_pressure)) > 0.0

        assert summary["pearson_r_mean"] > 0.999
        assert summary["pearson_r_median"] > 0.999
        assert 0.99 <= summary["rms_ratio_mean"] <= 1.01
        assert 0.99 <= summary["rms_ratio_median"] <= 1.01
        assert summary["rmse_median"] < 1e-3
        assert summary["max_abs_diff_max"] < 1e-2
        assert 0.99 <= summary["peak_ratio_median"] <= 1.01

        for row, metrics in trace_metrics.items():
            assert metrics["pearson_r"] > 0.999, f"sensor row {row} correlation too low"
            assert 0.99 <= metrics["peak_ratio"] <= 1.01, f"sensor row {row} peak ratio out of range"
