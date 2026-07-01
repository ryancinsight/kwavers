"""Regression test for the vendored checkpointing example."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from parity_test_utils import (
    assert_decodable_nonblank_png,
    load_example_module,
    report_metric_value,
)


run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow checkpointing tests"


def _load_module():
    return load_example_module("checkpointing_compare.py")


def _metric_bool(text: str, label: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            payload = stripped.split(":", 1)[1].strip()
            if payload == "True":
                return True
            if payload == "False":
                return False
            raise AssertionError(f"metric {label} is not boolean: {payload}")
    raise AssertionError(f"missing metric line: {label}")


def _metric_tuple(text: str, label: str) -> tuple[int, int]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            payload = stripped.split(":", 1)[1].strip()
            left, right = payload.strip("()").split(",", 1)
            return (int(left), int(right))
    raise AssertionError(f"missing metric line: {label}")


def _assert_report_contract(module, text: str) -> None:
    contract = module.CHECKPOINT_CONTRACT
    expected_shape = module.expected_sensor_shape()

    assert "checkpoint_status: PASS" in text
    assert (
        report_metric_value(text, "max_absolute_error")
        == contract["max_absolute_error"]
    )
    assert _metric_bool(text, "bit_exact") is contract["bit_exact"]
    assert _metric_tuple(text, "reference_shape") == expected_shape
    assert _metric_tuple(text, "resumed_shape") == expected_shape
    assert report_metric_value(text, "reference_runtime_s") > 0.0
    assert report_metric_value(text, "checkpoint_runtime_s") > 0.0
    assert report_metric_value(text, "resume_runtime_s") > 0.0
    assert (
        report_metric_value(text, "checkpoint_size_bytes")
        >= contract["min_checkpoint_size_bytes"]
    )
    assert _metric_bool(text, "checkpoint_deleted") is contract["checkpoint_deleted"]
    assert report_metric_value(text, "nt") == expected_shape[1]
    assert report_metric_value(text, "split_step") == module.SPLIT


def test_current_checkpointing_artifacts_match_contract():
    module = _load_module()
    text = module.METRICS_PATH.read_text(encoding="utf-8")

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    _assert_report_contract(module, text)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


@pytest.mark.skipif(not run_slow, reason=slow_reason)
@pytest.mark.timeout(180)
class TestCheckpointingParity:
    """Exact parity coverage for PSTD checkpoint save/resume."""

    def test_checkpointing_run_comparison(self):
        module = _load_module()
        result = module.run_comparison()
        report_lines = module.build_report_lines(result)

        assert result["status"] == "PASS"
        assert all(result["contract_checks"].values())
        assert report_lines[0] == "checkpoint_status: PASS"
        assert any(
            line.startswith(
                "max_absolute_error: "
                f"{module.CHECKPOINT_CONTRACT['max_absolute_error']:.6e}"
            )
            for line in report_lines
        )
        assert any(
            line == f"bit_exact: {module.CHECKPOINT_CONTRACT['bit_exact']}"
            for line in report_lines
        )
        assert Path(result["metrics_path"]).exists()
        assert Path(result["plot_path"]).exists()
        assert_decodable_nonblank_png(Path(result["plot_path"]))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
