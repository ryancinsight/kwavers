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
from parity_test_utils import assert_decodable_nonblank_png, report_metric_value


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


def _assert_metric_contract(metric, thresholds):
    if "pearson_r" in thresholds:
        assert metric["pearson_r"] >= thresholds["pearson_r"]
    if "max_abs_diff" in thresholds:
        assert metric["max_abs_diff"] < thresholds["max_abs_diff"]
    if "rms_ratio_min" in thresholds:
        assert thresholds["rms_ratio_min"] <= metric["rms_ratio"]
    if "rms_ratio_max" in thresholds:
        assert metric["rms_ratio"] <= thresholds["rms_ratio_max"]
    if "rmse" in thresholds:
        assert metric["rmse"] < thresholds["rmse"]
    if "psnr_db" in thresholds:
        assert metric["psnr_db"] > thresholds["psnr_db"]


def _assert_report_section_contract(text: str, section: str, thresholds):
    if "pearson_r" in thresholds:
        assert report_metric_value(text, "pearson_r", section) >= thresholds["pearson_r"]
    if "max_abs_diff" in thresholds:
        assert report_metric_value(text, "max_abs_diff", section) < thresholds["max_abs_diff"]
    if "rms_ratio_min" in thresholds or "rms_ratio_max" in thresholds:
        rms_ratio = report_metric_value(text, "rms_ratio", section)
        if "rms_ratio_min" in thresholds:
            assert thresholds["rms_ratio_min"] <= rms_ratio
        if "rms_ratio_max" in thresholds:
            assert rms_ratio <= thresholds["rms_ratio_max"]
    if "rmse" in thresholds:
        assert report_metric_value(text, "rmse", section) < thresholds["rmse"]
    if "psnr_db" in thresholds:
        assert report_metric_value(text, "psnr_db", section) > thresholds["psnr_db"]


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS

    assert "parity_status: PASS" in text
    _assert_report_section_contract(text, "source mask:", thresholds["source_mask"])
    _assert_report_section_contract(
        text,
        "source weighted mask:",
        thresholds["source_weighted_mask"],
    )
    _assert_report_section_contract(
        text,
        "distributed source signal:",
        thresholds["source_signal"],
    )
    _assert_report_section_contract(text, "p_rms:", thresholds["p_rms"])
    _assert_report_section_contract(text, "p_max:", thresholds["p_max"])


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_at_array_as_source_artifacts_match_thresholds():
    module = _load_example_module()
    text = module.METRICS_PATH.read_text(encoding="utf-8")

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


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

        for metric_name, thresholds in module.PARITY_THRESHOLDS.items():
            _assert_metric_contract(metrics[metric_name], thresholds)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
