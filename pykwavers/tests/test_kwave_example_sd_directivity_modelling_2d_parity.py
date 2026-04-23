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

        assert matrix_metrics["pearson_r"] > 0.99
        assert matrix_metrics["psnr_db"] > 30.0
        assert trace_summary["pearson_r_min"] > 0.99
        assert abs(trace_summary["rms_ratio_mean"] - 1.0) < 1e-2
        assert directivity["pearson_r"] > 0.99
        assert abs(directivity["rms_ratio"] - 1.0) < 1e-2
        assert abs(directivity["peak_ratio"] - 1.0) < 1e-2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
