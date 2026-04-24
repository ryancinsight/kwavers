"""Regression test for the vendored checkpointing example."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow checkpointing tests"


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


@pytest.mark.skipif(not run_slow, reason=slow_reason)
@pytest.mark.timeout(180)
class TestCheckpointingParity:
    """Exact parity coverage for PSTD checkpoint save/resume."""

    def test_checkpointing_run_comparison(self):
        module = _load_module("checkpointing_compare", "checkpointing_compare.py")
        result = module.run_comparison()
        metrics = result["metrics"]
        reference = result["reference"]
        checkpoint = result["checkpoint"]
        resumed = result["resumed"]
        report_lines = module.build_report_lines(result)

        expected_shape = (module.NX * module.NY * module.NZ, module.NT)

        assert result["status"] == "PASS"
        assert metrics["bit_exact"] is True
        assert metrics["max_absolute_error"] == 0.0
        assert metrics["reference_shape"] == expected_shape
        assert metrics["resumed_shape"] == expected_shape
        assert reference["shape"] == expected_shape
        assert resumed["shape"] == expected_shape
        assert reference["runtime_s"] > 0.0
        assert checkpoint["checkpoint_runtime_s"] > 0.0
        assert checkpoint["resume_runtime_s"] > 0.0
        assert checkpoint["checkpoint_size_bytes"] > 0
        assert checkpoint["checkpoint_deleted"] is True
        assert report_lines[0] == "checkpoint_status: PASS"
        assert any(line.startswith("max_absolute_error: 0.000000e+00") for line in report_lines)
        assert any(line == "bit_exact: True" for line in report_lines)
        assert Path(result["metrics_path"]).exists()
        assert Path(result["plot_path"]).exists()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
