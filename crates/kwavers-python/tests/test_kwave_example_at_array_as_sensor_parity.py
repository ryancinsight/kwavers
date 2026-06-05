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

        assert mask_metrics["pearson_r"] > 0.99999
        assert weighted_mask_metrics["pearson_r"] > 0.99999
        assert raw_matrix_metrics["pearson_r"] > 0.98
        assert raw_matrix_metrics["psnr_db"] > 30.0
        assert combined_matrix_metrics["pearson_r"] > 0.99
        assert combined_matrix_metrics["psnr_db"] > 30.0
        assert trace_summary["pearson_r_min"] > 0.99
        assert trace_summary["pearson_r_mean"] > 0.99

        for metrics in trace_metrics.values():
            assert metrics["pearson_r"] > 0.99
            assert abs(metrics["rms_ratio"] - 1.0) < 3e-2
            assert abs(metrics["peak_ratio"] - 1.0) < 1.5e-1
            assert metrics["rmse"] < 2e-2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
