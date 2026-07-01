#!/usr/bin/env python3
"""
Direct k-wave-python example parity tests for pykwavers.

This suite exercises exact example reproductions against the vendored
k-wave-python examples and asserts on metrics, not only on successful execution.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from conftest import requires_kwave
from parity_test_utils import (
    assert_decodable_nonblank_png,
    load_example_module,
    report_metric_value,
)


skip_kwave = os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1"
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow k-wave-python tests"


def _load_example_module():
    return load_example_module("compare_pr_3D_FFT_planar_sensor.py")


def _assert_report_summary_contract(text: str, thresholds: dict[str, dict[str, float]]):
    summary = thresholds["summary"]
    assert report_metric_value(text, "pearson_r_mean") > summary["pearson_r_mean"]
    assert report_metric_value(text, "pearson_r_median") > summary["pearson_r_median"]

    rms_mean = report_metric_value(text, "rms_ratio_mean")
    assert summary["rms_ratio_mean_min"] <= rms_mean <= summary["rms_ratio_mean_max"]

    rms_median = report_metric_value(text, "rms_ratio_median")
    assert summary["rms_ratio_median_min"] <= rms_median <= summary["rms_ratio_median_max"]

    assert report_metric_value(text, "rmse_median") < summary["rmse_median"]
    assert report_metric_value(text, "max_abs_diff") < summary["max_abs_diff_max"]

    peak_median = report_metric_value(text, "peak_ratio_median")
    assert summary["peak_ratio_median_min"] <= peak_median <= summary["peak_ratio_median_max"]


def _report_trace_metrics(text: str) -> dict[int, dict[str, float]]:
    trace_re = re.compile(
        r"row=(?P<row>\d+):\s+pearson_r=(?P<pearson_r>[0-9.eE+-]+)\s+"
        r"rms_ratio=(?P<rms_ratio>[0-9.eE+-]+)\s+"
        r"rmse=(?P<rmse>[0-9.eE+-]+)\s+"
        r"peak_ratio=(?P<peak_ratio>[0-9.eE+-]+)"
    )
    traces = {}
    for match in trace_re.finditer(text):
        row = int(match.group("row"))
        traces[row] = {
            name: float(match.group(name))
            for name in ("pearson_r", "rms_ratio", "rmse", "peak_ratio")
        }
    assert traces
    return traces


def _assert_report_trace_contract(text: str, thresholds: dict[str, dict[str, float]]):
    trace = thresholds["trace"]
    for row, metrics in _report_trace_metrics(text).items():
        assert metrics["pearson_r"] > trace["pearson_r"], row
        assert trace["peak_ratio_min"] <= metrics["peak_ratio"] <= trace["peak_ratio_max"], row


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_current_pr_3d_fft_planar_sensor_artifacts_match_thresholds():
    """Current PR 3-D FFT report and PNG satisfy the driver-owned contract."""
    module = _load_example_module()

    assert module.METRICS_PATH.exists()
    assert module.PRESSURE_FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")

    assert "parity_status: PASS" in text
    _assert_report_summary_contract(text, module.PARITY_THRESHOLDS)
    _assert_report_trace_contract(text, module.PARITY_THRESHOLDS)
    assert_decodable_nonblank_png(module.PRESSURE_FIGURE_PATH)


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestKWaveExampleParity:
    """Metrics-based parity coverage for vendored k-wave-python examples."""

    def test_pr_3d_fft_planar_sensor_metrics(self):
        """Exact planar-sensor initial-pressure example matches on trace metrics."""
        module = _load_example_module()
        result = module.run_comparison()

        kw_pressure = np.asarray(result["kwave"]["pressure"], dtype=np.float64)
        py_pressure = np.asarray(result["pykwavers"]["pressure"], dtype=np.float64)
        summary = result["summary"]
        trace_metrics = result["trace_metrics"]

        assert kw_pressure.shape == py_pressure.shape, (
            f"shape mismatch: {kw_pressure.shape} != {py_pressure.shape}"
        )
        assert np.all(np.isfinite(kw_pressure))
        assert np.all(np.isfinite(py_pressure))
        assert np.max(np.abs(kw_pressure)) > 0.0
        assert np.max(np.abs(py_pressure)) > 0.0

        assert summary["n_sensors"] == kw_pressure.shape[0]
        assert len(trace_metrics) == 3
        checks = module.evaluate_parity_contract(result)
        assert all(checks.values()), checks
