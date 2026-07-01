#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for the 3D focused bowl example.
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
    module_path = root / "examples" / "at_focused_bowl_3D_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("at_focused_bowl_3D_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_report_contract(module, result):
    thr = module.PARITY_THRESHOLDS
    summary = result["summary"]
    source_metrics = result["source_metrics"]

    assert summary["pearson_r"] >= thr["pearson_r"]
    assert thr["rms_ratio_min"] <= summary["rms_ratio"] <= thr["rms_ratio_max"]
    assert summary["psnr_db"] >= thr["psnr_db"]

    assert source_metrics["pearson_r"] >= thr["source_pearson_r"]
    assert (
        thr["source_rms_ratio_min"]
        <= source_metrics["rms_ratio"]
        <= thr["source_rms_ratio_max"]
    )
    assert (
        thr["source_peak_ratio_min"]
        <= source_metrics["peak_ratio"]
        <= thr["source_peak_ratio_max"]
    )
    assert source_metrics["rmse"] <= thr["source_rmse_max"]


def _assert_report_text_contract(module, text: str):
    thr = module.PARITY_THRESHOLDS
    source_section = "source weights (physical interior): k-wave-python vs pykwavers"

    assert "parity_status: PASS" in text
    assert report_metric_value(text, "pearson_r") >= thr["pearson_r"]
    rms_ratio = report_metric_value(text, "rms_ratio")
    assert thr["rms_ratio_min"] <= rms_ratio <= thr["rms_ratio_max"]
    assert report_metric_value(text, "psnr_db") >= thr["psnr_db"]

    assert (
        report_metric_value(text, "pearson_r", source_section)
        >= thr["source_pearson_r"]
    )
    source_rms = report_metric_value(text, "rms_ratio", source_section)
    source_peak = report_metric_value(text, "peak_ratio", source_section)
    assert thr["source_rms_ratio_min"] <= source_rms <= thr["source_rms_ratio_max"]
    assert thr["source_peak_ratio_min"] <= source_peak <= thr["source_peak_ratio_max"]
    assert report_metric_value(text, "rmse", source_section) <= thr["source_rmse_max"]


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_at_focused_bowl_3d_artifacts_match_thresholds():
    module = _load_example_module()
    text = module.METRICS_PATH.read_text(encoding="utf-8")

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    _assert_report_text_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityAtFocusedBowl3D:
    """Metrics-based parity coverage for the focused bowl example."""

    def test_at_focused_bowl_3d_metrics(self):
        module = _load_example_module()
        result = module.run_comparison()

        kw_amp = np.asarray(result["kwave"]["amp_axial"], dtype=np.float64)
        py_amp = np.asarray(result["pykwavers"]["amp_axial"], dtype=np.float64)
        kw_source = np.asarray(result["kwave"]["source_weights"], dtype=np.float64)
        py_source = np.asarray(result["pykwavers"]["source_weights"], dtype=np.float64)

        assert kw_amp.shape == py_amp.shape
        assert kw_source.shape == py_source.shape
        assert np.all(np.isfinite(kw_amp))
        assert np.all(np.isfinite(py_amp))
        assert np.all(np.isfinite(kw_source))
        assert np.all(np.isfinite(py_source))
        assert np.max(np.abs(kw_amp)) > 0.0
        assert np.max(np.abs(py_amp)) > 0.0
        assert np.max(np.abs(kw_source)) > 0.0
        assert np.max(np.abs(py_source)) > 0.0

        _assert_report_contract(module, result)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
