#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for the 2D focussed detector example.
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
    module_path = root / "examples" / "sd_focussed_detector_2D_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("sd_focussed_detector_2D_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParitySDFocussedDetector2D:
    """Metrics-based parity coverage for the 2D focussed detector example."""

    def test_sd_focussed_detector_2d_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kw_on = np.asarray(result["kwave"]["on_axis"]["trace"], dtype=np.float64)
        kw_off = np.asarray(result["kwave"]["off_axis"]["trace"], dtype=np.float64)
        py_on = np.asarray(result["pykwavers"]["on_axis"]["trace"], dtype=np.float64)
        py_off = np.asarray(result["pykwavers"]["off_axis"]["trace"], dtype=np.float64)
        trace_metrics = result["trace_metrics"]
        directivity = result["directivity_metrics"]

        assert kw_on.shape == py_on.shape
        assert kw_off.shape == py_off.shape
        assert kw_on.ndim == 1
        assert kw_off.ndim == 1
        assert np.all(np.isfinite(kw_on))
        assert np.all(np.isfinite(py_on))
        assert np.all(np.isfinite(kw_off))
        assert np.all(np.isfinite(py_off))
        assert np.max(np.abs(kw_on)) > 0.0
        assert np.max(np.abs(py_on)) > 0.0
        assert np.max(np.abs(kw_off)) > 0.0
        assert np.max(np.abs(py_off)) > 0.0

        for tag in ("on_axis", "off_axis"):
            metrics = trace_metrics[tag]
            assert metrics["pearson_r"] > 0.999
            assert abs(metrics["rms_ratio"] - 1.0) < 1e-2
            assert metrics["rmse"] < 1e-2
            assert abs(metrics["peak_ratio"] - 1.0) < 1e-2

        assert directivity["kwave_ratio"] > 1.0
        assert directivity["pykwavers_ratio"] > 1.0
        assert directivity["relative_error"] < 1e-2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
