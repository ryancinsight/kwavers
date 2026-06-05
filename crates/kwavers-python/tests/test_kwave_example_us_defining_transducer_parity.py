#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `us_defining_transducer`.
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
    module_path = root / "examples" / "us_defining_transducer_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("us_defining_transducer_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityUsDefiningTransducer:
    """Metrics-based parity coverage for the defining transducer example."""

    def test_us_defining_transducer_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kwave = result["kwave"]
        pykwavers = result["pykwavers"]

        kw_pressure = np.asarray(kwave["pressure"], dtype=np.float64)
        py_pressure = np.asarray(pykwavers["pressure"], dtype=np.float64)
        assert kw_pressure.shape == py_pressure.shape
        assert kw_pressure.shape == (3, 452)
        assert int(kwave["time_steps"]) == 452
        assert int(pykwavers["time_steps"]) == 452
        assert np.max(np.abs(kw_pressure)) > 0.0
        assert np.max(np.abs(py_pressure)) > 0.0

        for idx in range(kw_pressure.shape[0]):
            metrics = module.compute_trace_metrics(kw_pressure[idx], py_pressure[idx])
            assert metrics["pearson_r"] > 0.99
            assert 0.99 <= metrics["rms_ratio"] <= 1.15
            assert 0.99 <= metrics["peak_ratio"] <= 1.13
            assert metrics["rmse"] < 1.2e4
            assert metrics["max_abs_diff"] < 5.0e4

        report_lines = module.build_report_lines(kwave, pykwavers)
        assert report_lines[0] == "parity_status: PASS"
        assert any(line == "sensor_1: PASS" for line in report_lines)
        assert any(line == "sensor_2: PASS" for line in report_lines)
        assert any(line == "sensor_3: PASS" for line in report_lines)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
