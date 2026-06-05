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

        assert metrics1["pearson_r"] > 0.99
        assert metrics2["pearson_r"] > 0.99
        assert abs(metrics1["rms_ratio"] - 1.0) < 1e-5
        assert abs(metrics2["rms_ratio"] - 1.0) < 1e-4
        assert metrics1["psnr_db"] > 110.0
        assert metrics2["psnr_db"] > 100.0

        kw_dir_ratio = float(np.abs(kw1_trace).max()) / (float(np.abs(kw2_trace).max()) + 1e-30)
        pkw_dir_ratio = float(np.abs(pkw1_trace).max()) / (float(np.abs(pkw2_trace).max()) + 1e-30)
        assert kw_dir_ratio > 1.0
        assert pkw_dir_ratio > 1.0
        assert report_text.startswith("sd_focussed_detector_3D parity metrics")
        assert "parity_status: PASS" in report_text
        assert "src1_on_axis: PASS" in report_text
        assert "src2_off_axis: PASS" in report_text
        assert "directivity_test:" in report_text
        assert "status = PASS" in report_text


if __name__ == "__main__":
    pytest.main(["-v", __file__])
