#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `at_array_as_source`.
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
    module_path = root / "examples" / "at_array_as_source_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("at_array_as_source_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityAtArrayAsSource:
    """Metrics-based parity coverage for the array-as-source example."""

    def test_at_array_as_source_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kwave = result["kwave"]
        pykwavers = result["pykwavers"]
        layout = result["layout"]
        metrics = result["metrics"]

        source_mask_kw = np.asarray(layout["source_binary_mask"], dtype=bool)
        source_mask_py = np.asarray(pykwavers["source_binary_mask"], dtype=bool)
        source_weighted_mask_kw = np.asarray(layout["source_weighted_mask"], dtype=np.float64)
        source_weighted_mask_py = np.asarray(pykwavers["source_weighted_mask"], dtype=np.float64)
        source_signal_kw = np.asarray(layout["source_signal_kw"], dtype=np.float64)
        source_signal_py = np.asarray(layout["source_signal_py"], dtype=np.float64)
        kw_p_max = np.asarray(kwave["p_max"], dtype=np.float64)
        py_p_max = np.asarray(pykwavers["p_max"], dtype=np.float64)
        kw_p_rms = np.asarray(kwave["p_rms"], dtype=np.float64)
        py_p_rms = np.asarray(pykwavers["p_rms"], dtype=np.float64)

        assert source_mask_kw.shape == source_mask_py.shape
        assert source_weighted_mask_kw.shape == source_weighted_mask_py.shape
        assert source_signal_kw.shape == source_signal_py.shape
        assert kw_p_max.shape == py_p_max.shape
        assert kw_p_rms.shape == py_p_rms.shape

        assert np.array_equal(source_mask_kw, source_mask_py)
        assert np.allclose(source_weighted_mask_kw, source_weighted_mask_py, rtol=1e-12, atol=1e-12)
        assert np.allclose(source_signal_kw, source_signal_py, rtol=1e-12, atol=1e-12)
        assert np.allclose(kw_p_max, py_p_max, rtol=1e-5, atol=1e-5)
        assert np.allclose(kw_p_rms, py_p_rms, rtol=1e-3, atol=3e-4)

        assert metrics["source_mask"]["pearson_r"] > 0.999999
        assert metrics["source_weighted_mask"]["pearson_r"] > 0.999999
        assert metrics["source_signal"]["pearson_r"] > 0.999999
        assert metrics["p_max"]["pearson_r"] > 0.999999
        assert metrics["p_rms"]["pearson_r"] > 0.999999
        assert metrics["source_weighted_mask"]["max_abs_diff"] < 1e-12
        assert metrics["source_signal"]["max_abs_diff"] < 1e-12
        assert metrics["p_max"]["max_abs_diff"] < 1e-5
        assert metrics["p_rms"]["max_abs_diff"] < 2e-4
        assert abs(metrics["p_max"]["rms_ratio"] - 1.0) < 1e-6
        assert abs(metrics["p_max"]["rmse"]) < 1e-5
        assert abs(metrics["p_rms"]["rms_ratio"] - 1.0) < 1e-3
        assert metrics["p_rms"]["rmse"] < 1e-4
        assert metrics["p_rms"]["psnr_db"] > 70.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
