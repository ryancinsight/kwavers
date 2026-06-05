#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `at_linear_array_transducer`.
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
    module_path = root / "examples" / "at_linear_array_transducer_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("at_linear_array_transducer_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityAtLinearArrayTransducer:
    """Metrics-based parity coverage for the linear-array transducer example."""

    def test_at_linear_array_transducer_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kwave = result["kwave"]
        pykwavers = result["pykwavers"]
        metrics = result["metrics"]

        source_mask_kw = np.asarray(kwave["source_binary_mask"], dtype=bool)
        source_mask_py = np.asarray(pykwavers["source_binary_mask"], dtype=bool)
        source_weighted_mask_kw = np.asarray(kwave["source_weighted_mask"], dtype=np.float64)
        source_weighted_mask_py = np.asarray(pykwavers["source_weighted_mask"], dtype=np.float64)
        kw_p_max = np.asarray(kwave["p_max"], dtype=np.float64)
        py_p_max = np.asarray(pykwavers["p_max"], dtype=np.float64)
        report_lines = module.build_report_lines(result)

        assert source_mask_kw.shape == source_mask_py.shape
        assert source_weighted_mask_kw.shape == source_weighted_mask_py.shape
        assert kw_p_max.shape == py_p_max.shape

        assert np.array_equal(source_mask_kw, source_mask_py)
        assert np.allclose(source_weighted_mask_kw, source_weighted_mask_py, rtol=1e-12, atol=1e-12)
        assert np.all(np.isfinite(kw_p_max))
        assert np.all(np.isfinite(py_p_max))

        assert metrics["source_mask"]["kwave_active_cells"] == metrics["source_mask"]["pykwavers_active_cells"]
        assert metrics["source_mask"]["active_cell_ratio"] == 1.0
        assert metrics["source_mask"]["iou"] == 1.0
        assert metrics["source_mask"]["dice"] == 1.0
        assert metrics["source_weighted_mask"]["pearson_r"] > 0.999999999999
        assert metrics["source_weighted_mask"]["max_abs_diff"] < 1e-12
        assert abs(metrics["source_weighted_mask"]["rms_ratio"] - 1.0) < 1e-12
        assert metrics["source_weighted_mask"]["peak_ratio"] == 1.0
        assert metrics["source_weighted_mask"]["psnr_db"] > 300.0
        assert metrics["p_max"]["pearson_r"] > 0.9999999999
        assert abs(metrics["p_max"]["rms_ratio"] - 1.0) < 1e-5
        assert abs(metrics["p_max"]["peak_ratio"] - 1.0) < 1e-4
        assert metrics["p_max"]["psnr_db"] > 100.0
        assert any(line.strip() == f"source_mode: {module.SOURCE_MODE}" for line in report_lines)
        assert any(line.strip() == f"compatibility_mode: {module.COMPATIBILITY_MODE}" for line in report_lines)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
