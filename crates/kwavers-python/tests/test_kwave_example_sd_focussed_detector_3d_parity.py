#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `sd_focussed_detector_3D`.
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
    module_path = root / "examples" / "sd_focussed_detector_3D_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("sd_focussed_detector_3D_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_trace_contract(metrics, thresholds):
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert metrics["psnr_db"] >= thresholds["psnr_db"]


def _assert_report_contract(module, text: str):
    assert text.startswith("sd_focussed_detector_3D parity metrics")
    assert "parity_status: PASS" in text
    assert "directivity_test:" in text
    assert "status = PASS" in text

    thresholds = module.PARITY_THRESHOLDS
    for section, threshold_key in (
        ("src1_on_axis: PASS", "src1_on_axis"),
        ("src2_off_axis: PASS", "src2_off_axis"),
    ):
        trace_thresholds = thresholds[threshold_key]
        assert report_metric_value(text, "pearson_r", section) >= trace_thresholds["pearson_r"]
        rms_ratio = report_metric_value(text, "rms_ratio", section)
        assert trace_thresholds["rms_ratio_min"] <= rms_ratio
        assert rms_ratio <= trace_thresholds["rms_ratio_max"]
        assert report_metric_value(text, "psnr_db", section) >= trace_thresholds["psnr_db"]

    directivity_thresholds = thresholds["directivity"]
    assert (
        report_metric_value(text, "kwave_on_off_ratio", "directivity_test:")
        > directivity_thresholds["ratio_min"]
    )
    assert (
        report_metric_value(text, "pykwavers_on_off_ratio", "directivity_test:")
        > directivity_thresholds["ratio_min"]
    )


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_sd_focussed_detector_3d_artifacts_match_thresholds():
    module = _load_example_module()

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    assert module.DIR_FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)
    assert_decodable_nonblank_png(module.DIR_FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParitySDFocussedDetector3D:
    """Metrics-based parity coverage for the focused detector example."""

    def test_sd_focussed_detector_3d_metrics(self):
        module = _load_example_module()
        kgrid, medium, bowl_mask_bool, signal_1d, dt, nt = module.build_config()

        kw1 = module.run_kwave(
            "src1",
            kgrid,
            medium,
            bowl_mask_bool,
            signal_1d,
            module.SRC1_IX,
            module.SRC1_IY,
            module.SRC1_IZ,
            use_gpu=False,
        )
        kw2 = module.run_kwave(
            "src2",
            kgrid,
            medium,
            bowl_mask_bool,
            signal_1d,
            module.SRC2_IX,
            module.SRC2_IY,
            module.SRC2_IZ,
            use_gpu=False,
        )
        pkw1 = module.run_pykwavers(
            "src1",
            bowl_mask_bool,
            signal_1d,
            module.SRC1_IX,
            module.SRC1_IY,
            module.SRC1_IZ,
            dt,
            nt,
        )
        pkw2 = module.run_pykwavers(
            "src2",
            bowl_mask_bool,
            signal_1d,
            module.SRC2_IX,
            module.SRC2_IY,
            module.SRC2_IZ,
            dt,
            nt,
        )

        kw1_trace = np.asarray(kw1["trace"], dtype=np.float64)
        kw2_trace = np.asarray(kw2["trace"], dtype=np.float64)
        pkw1_trace = np.asarray(pkw1["trace"], dtype=np.float64)
        pkw2_trace = np.asarray(pkw2["trace"], dtype=np.float64)
        metrics1 = module.compute_image_metrics(kw1_trace, pkw1_trace)
        metrics2 = module.compute_image_metrics(kw2_trace, pkw2_trace)
        report_text = Path(module.METRICS_PATH).read_text(encoding="utf-8")

        assert kw1_trace.shape == pkw1_trace.shape == (nt,)
        assert kw2_trace.shape == pkw2_trace.shape == (nt,)
        assert np.max(np.abs(kw1_trace)) > 0.0
        assert np.max(np.abs(kw2_trace)) > 0.0
        assert np.max(np.abs(pkw1_trace)) > 0.0
        assert np.max(np.abs(pkw2_trace)) > 0.0

        _assert_trace_contract(metrics1, module.PARITY_THRESHOLDS["src1_on_axis"])
        _assert_trace_contract(metrics2, module.PARITY_THRESHOLDS["src2_off_axis"])

        kw_dir_ratio = float(np.abs(kw1_trace).max()) / (float(np.abs(kw2_trace).max()) + 1e-30)
        pkw_dir_ratio = float(np.abs(pkw1_trace).max()) / (float(np.abs(pkw2_trace).max()) + 1e-30)
        assert kw_dir_ratio > module.PARITY_THRESHOLDS["directivity"]["ratio_min"]
        assert pkw_dir_ratio > module.PARITY_THRESHOLDS["directivity"]["ratio_min"]
        _assert_report_contract(module, report_text)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
