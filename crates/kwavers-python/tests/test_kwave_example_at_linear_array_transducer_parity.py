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
from parity_test_utils import assert_decodable_nonblank_png, report_metric_value


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


def _assert_source_mask_contract(metrics, thresholds):
    assert metrics["kwave_active_cells"] == metrics["pykwavers_active_cells"]
    assert metrics["active_cell_ratio"] == thresholds["active_cell_ratio"]
    assert metrics["iou"] == thresholds["iou"]
    assert metrics["dice"] == thresholds["dice"]


def _assert_weighted_mask_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert metrics["max_abs_diff"] < thresholds["max_abs_diff"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert metrics["peak_ratio"] == thresholds["peak_ratio"]
    assert metrics["psnr_db"] > thresholds["psnr_db"]


def _assert_p_max_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert metrics["psnr_db"] > thresholds["psnr_db"]


def _assert_report_contract(module, text: str):
    thresholds = module.PARITY_THRESHOLDS

    assert "parity_status: PASS" in text
    assert report_metric_value(text, "active_cell_ratio") == thresholds["source_mask"]["active_cell_ratio"]
    assert report_metric_value(text, "iou") == thresholds["source_mask"]["iou"]
    assert report_metric_value(text, "dice") == thresholds["source_mask"]["dice"]

    assert (
        report_metric_value(text, "pearson_r", "source weighted-mask parity")
        >= thresholds["source_weighted_mask"]["pearson_r"]
    )
    assert (
        report_metric_value(text, "max_abs_diff", "source weighted-mask parity")
        < thresholds["source_weighted_mask"]["max_abs_diff"]
    )
    weighted_rms = report_metric_value(text, "rms_ratio", "source weighted-mask parity")
    assert thresholds["source_weighted_mask"]["rms_ratio_min"] <= weighted_rms
    assert weighted_rms <= thresholds["source_weighted_mask"]["rms_ratio_max"]
    assert (
        report_metric_value(text, "peak_ratio", "source weighted-mask parity")
        == thresholds["source_weighted_mask"]["peak_ratio"]
    )
    assert (
        report_metric_value(text, "psnr_db", "source weighted-mask parity")
        > thresholds["source_weighted_mask"]["psnr_db"]
    )

    assert (
        report_metric_value(text, "pearson_r", "p_max field parity")
        >= thresholds["p_max"]["pearson_r"]
    )
    p_max_rms = report_metric_value(text, "rms_ratio", "p_max field parity")
    assert thresholds["p_max"]["rms_ratio_min"] <= p_max_rms
    assert p_max_rms <= thresholds["p_max"]["rms_ratio_max"]
    assert (
        report_metric_value(text, "psnr_db", "p_max field parity")
        > thresholds["p_max"]["psnr_db"]
    )


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_at_linear_array_transducer_artifacts_match_thresholds():
    module = _load_example_module()
    text = module.METRICS_PATH.read_text(encoding="utf-8")

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


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

        thresholds = module.PARITY_THRESHOLDS
        _assert_source_mask_contract(metrics["source_mask"], thresholds["source_mask"])
        _assert_weighted_mask_contract(
            metrics["source_weighted_mask"],
            thresholds["source_weighted_mask"],
        )
        _assert_p_max_contract(metrics["p_max"], thresholds["p_max"])
        assert any(line.strip() == f"source_mode: {module.SOURCE_MODE}" for line in report_lines)
        assert any(line.strip() == f"compatibility_mode: {module.COMPATIBILITY_MODE}" for line in report_lines)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
