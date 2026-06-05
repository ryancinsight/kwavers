#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `ivp_photoacoustic_waveforms`.
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
    module_path = root / "examples" / "ivp_photoacoustic_waveforms_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("ivp_photoacoustic_waveforms_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityIvpPhotoacousticWaveforms:
    """Metrics-based parity coverage for the IVP photoacoustic waveform example."""

    def test_ivp_photoacoustic_waveforms_metrics(self):
        module = _load_example_module()
        p0 = module.build_p0()
        kw = module.run_kwave(p0)
        pkw = module.run_pykwavers(p0)

        kw_trace = np.asarray(kw["trace"], dtype=np.float64)
        pkw_trace = np.asarray(pkw["trace"], dtype=np.float64)
        metrics = module.compute_image_metrics(kw_trace, pkw_trace)
        report_text = Path(module.METRICS_PATH).read_text(encoding="utf-8")

        assert kw_trace.shape == pkw_trace.shape
        assert kw_trace.shape == (module.NT,)
        assert np.max(np.abs(kw_trace)) > 0.0
        assert np.max(np.abs(pkw_trace)) > 0.0
        assert metrics["pearson_r"] > 0.98
        assert abs(metrics["rms_ratio"] - 1.0) < 1e-6
        assert metrics["psnr_db"] > 24.0
        assert abs(metrics["peak_ratio"] - 1.0) < 1e-6
        assert "parity_status: PASS" in report_text
        assert "peak_ratio        = 1.000000" in report_text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
