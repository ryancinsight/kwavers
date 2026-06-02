#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for the circular piston example.
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
    module_path = root / "examples" / "at_circular_piston_3D_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("at_circular_piston_3D_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityAtCircularPiston3D:
    """Metrics-based parity coverage for the circular piston example."""

    def test_at_circular_piston_3d_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kw_amp = np.asarray(result["kwave"]["amp_on_axis"], dtype=np.float64)
        py_amp = np.asarray(result["pykwavers"]["amp_on_axis"], dtype=np.float64)
        analytical = np.asarray(result["analytical"]["amp_on_axis"], dtype=np.float64)
        summary = result["summary"]
        source_metrics = result["source_metrics"]
        reference_metrics = result["reference_metrics"]

        assert kw_amp.shape == py_amp.shape
        assert kw_amp.shape == analytical.shape
        assert np.all(np.isfinite(kw_amp))
        assert np.all(np.isfinite(py_amp))
        assert np.all(np.isfinite(analytical))
        assert np.max(np.abs(kw_amp)) > 0.0
        assert np.max(np.abs(py_amp)) > 0.0
        assert np.max(np.abs(analytical)) > 0.0

        assert summary["pearson_r"] > 0.99999
        assert abs(summary["rms_ratio"] - 1.0) < 1e-4
        assert abs(summary["peak_ratio"] - 1.0) < 1e-4
        assert source_metrics["pearson_r"] > 0.99999
        assert abs(source_metrics["rms_ratio"] - 1.0) < 1e-4
        assert abs(source_metrics["peak_ratio"] - 1.0) < 1e-4

        assert reference_metrics["kwave"]["pearson_r"] > 0.9998
        assert reference_metrics["pykwavers"]["pearson_r"] > 0.9998
        assert abs(reference_metrics["kwave"]["rms_ratio"] - 1.0) < 1e-2
        assert abs(reference_metrics["pykwavers"]["rms_ratio"] - 1.0) < 1e-2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
