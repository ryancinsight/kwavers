#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for `us_bmode_phased_array`.
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
    module_path = root / "examples" / "us_bmode_phased_array_compare.py"
    examples_dir = str(root / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("us_bmode_phased_array_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParityUsbmodePhasedArray:
    """Metrics-based parity coverage for the phased-array B-mode example."""

    def test_us_bmode_phased_array_metrics(self):
        module = _load_example_module()
        seed = 20260401
        steering_angles = module.STEERING_ANGLES_QUICK
        cache_tag = f"quick_seed{seed}_kwcpu_pkwgpu"

        kgrid, medium, transducer, not_transducer, input_signal = module.build_reference_objects(seed)
        kw_scan_lines, kw_runtime = module.run_kwave_phased_array(
            medium,
            kgrid,
            not_transducer,
            steering_angles,
            False,
            cache_tag,
        )
        pkw_scan_lines, pkw_runtime, pkw_profile = module.run_pykwavers_phased_array(
            medium.sound_speed,
            medium.density,
            kgrid,
            transducer,
            not_transducer,
            input_signal,
            steering_angles,
            True,
            cache_tag,
        )

        kw_raw, kw_fund, kw_harm, kw_bmode_fund, kw_bmode_harm = module.post_process(
            kw_scan_lines.copy(),
            kgrid,
            medium,
            not_transducer,
            steering_angles,
        )
        pkw_raw, pkw_fund, pkw_harm, pkw_bmode_fund, pkw_bmode_harm = module.post_process(
            pkw_scan_lines.copy(),
            kgrid,
            medium,
            not_transducer,
            steering_angles,
        )

        kwave_bundle = {
            "runtime_s": kw_runtime,
            "raw": kw_raw,
            "fund": kw_fund,
            "harm": kw_harm,
            "bmode_fund": kw_bmode_fund,
            "bmode_harm": kw_bmode_harm,
        }
        pykwavers_bundle = {
            "runtime_s": pkw_runtime,
            "raw": pkw_raw,
            "fund": pkw_fund,
            "harm": pkw_harm,
            "bmode_fund": pkw_bmode_fund,
            "bmode_harm": pkw_bmode_harm,
            "gpu_profile": pkw_profile,
        }

        metrics_fund = module.compute_image_metrics(kw_bmode_fund, pkw_bmode_fund)
        metrics_harm = module.compute_image_metrics(kw_bmode_harm, pkw_bmode_harm)
        report_lines = module.build_report_lines(kwave_bundle, pykwavers_bundle, steering_angles)

        assert kw_scan_lines.shape == pkw_scan_lines.shape
        assert kw_raw.shape == pkw_raw.shape
        assert kw_fund.shape == pkw_fund.shape
        assert kw_harm.shape == pkw_harm.shape
        assert kw_bmode_fund.shape == pkw_bmode_fund.shape
        assert kw_bmode_harm.shape == pkw_bmode_harm.shape
        assert kw_scan_lines.shape[0] == 9
        assert kw_scan_lines.shape[1] > 0
        assert np.max(np.abs(kw_scan_lines)) > 0.0
        assert np.max(np.abs(pkw_scan_lines)) > 0.0
        assert np.max(np.abs(kw_bmode_fund)) > 0.0
        assert np.max(np.abs(pkw_bmode_fund)) > 0.0
        assert np.max(np.abs(kw_bmode_harm)) > 0.0
        assert np.max(np.abs(pkw_bmode_harm)) > 0.0

        assert metrics_fund["pearson_r"] > 0.999
        assert metrics_fund["rms_ratio"] > 1.0
        assert metrics_fund["rms_ratio"] < 1.01
        assert metrics_fund["psnr_db"] > 45.0
        assert metrics_harm["pearson_r"] > 0.996
        assert metrics_harm["rms_ratio"] > 1.03
        assert metrics_harm["rms_ratio"] < 1.04
        assert metrics_harm["psnr_db"] > 40.0
        assert report_lines[0] == "parity_status: PASS"
        assert any(line == "  status    = PASS" for line in report_lines)
        assert pkw_profile is not None
        assert pkw_profile["total_ns"].shape == (len(steering_angles),)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
