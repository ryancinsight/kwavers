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
from parity_test_utils import assert_decodable_nonblank_png, report_metric_value


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


def _assert_metrics_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert metrics["psnr_db"] >= thresholds["psnr_db"]
    assert thresholds["peak_ratio_min"] <= metrics["peak_ratio"]
    assert metrics["peak_ratio"] <= thresholds["peak_ratio_max"]


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS

    assert "parity_status: PASS" in text
    assert report_metric_value(text, "pearson_r") >= thresholds["pearson_r"]
    rms_ratio = report_metric_value(text, "rms_ratio")
    assert thresholds["rms_ratio_min"] <= rms_ratio
    assert rms_ratio <= thresholds["rms_ratio_max"]
    assert report_metric_value(text, "psnr_db") >= thresholds["psnr_db"]
    peak_ratio = report_metric_value(text, "peak_ratio")
    assert thresholds["peak_ratio_min"] <= peak_ratio
    assert peak_ratio <= thresholds["peak_ratio_max"]


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_ivp_photoacoustic_waveforms_artifacts_match_thresholds():
    module = _load_example_module()

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


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
        _assert_metrics_contract(metrics, module.PARITY_THRESHOLDS)
        _assert_report_contract(module, report_text)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
