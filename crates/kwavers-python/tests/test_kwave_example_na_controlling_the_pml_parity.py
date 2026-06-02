"""Parity regression for the vendored `na_controlling_the_pml` example."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from conftest import requires_kwave


skip_kwave = os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1"
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow k-wave-python tests"


def _load_module(module_name: str, file_name: str):
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    module_path = examples_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleNaControllingThePmlParity:
    """Exact parity checks for all four upstream PML configurations."""

    def test_na_controlling_the_pml_run_comparison(self):
        module = _load_module("na_controlling_the_pml_compare", "na_controlling_the_pml_compare.py")
        result = module.run_comparison()

        assert result["status"] == "PASS"
        assert result["waveform"]["status"] == "PASS"
        assert result["hdf5"]["status"] == "PASS"
        assert result["summary"]["pearson_r_min"] > 0.95
        assert result["summary"]["rms_ratio_min"] > 0.85
        assert result["summary"]["rms_ratio_max"] < 1.15
        assert result["summary"]["psnr_db_min"] > 20.0
        assert result["summary"]["max_abs_diff_max"] < 1.0
        assert result["hdf5_summary"]["status"] == "PASS"
        assert result["hdf5_summary"]["dataset_max_abs_diff_max"] == 0.0
        assert result["hdf5_summary"]["root_attr_mismatch_count"] == 0

        for config_name, case in result["cases"].items():
            metrics = case["metrics"]
            hdf5_metrics = case["hdf5"]
            assert metrics["pearson_r"] > 0.95, config_name
            assert 0.85 <= metrics["rms_ratio"] <= 1.15, config_name
            assert metrics["psnr_db"] > 20.0, config_name
            assert metrics["max_abs_diff"] < 1.0, config_name
            assert hdf5_metrics["status"] == "PASS", config_name
            assert hdf5_metrics["max_abs_diff"] == 0.0, config_name
            assert hdf5_metrics["missing_datasets"] == [], config_name
            assert hdf5_metrics["extra_datasets"] == [], config_name
            assert hdf5_metrics["root_attr_mismatches"] == {}, config_name
            for dataset_name, dataset_result in hdf5_metrics["dataset_results"].items():
                assert dataset_result["data_match"], (config_name, dataset_name)
                assert dataset_result["attrs_match"], (config_name, dataset_name)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
