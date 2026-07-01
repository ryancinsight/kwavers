#!/usr/bin/env python3
"""Direct cached parity tests for k-wave-python example drivers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from conftest import requires_kwave
from parity_test_utils import (
    assert_cached_example_artifacts,
    load_example_module,
    load_numeric_cache,
)


skip_kwave = os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1"


@dataclass(frozen=True)
class CachedParityCase:
    script: str
    kwave_cache_attr: str
    pykwavers_cache_attr: str
    arrays: tuple[str, ...]
    expected_shape: tuple[int, ...]
    metric_method: str = "compute_image_metrics"
    runtime_keys: tuple[str, ...] = ("runtime_s",)
    kwave_permutation_key: str | None = None


CASES = (
    CachedParityCase(
        script="at_focused_annular_array_3D_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("amp_axial",),
        expected_shape=(67,),
    ),
    CachedParityCase(
        script="at_focused_annular_array_3D_full_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("amp_axial",),
        expected_shape=(67,),
    ),
    CachedParityCase(
        script="us_beam_patterns_compare.py",
        kwave_cache_attr="KWAVE_CACHE",
        pykwavers_cache_attr="PKWAV_CACHE",
        arrays=("p_rms", "p_max"),
        expected_shape=(88, 44),
    ),
    CachedParityCase(
        script="na_modelling_absorption_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(2, 1601),
    ),
    CachedParityCase(
        script="ivp_3D_simulation_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(4096, 444),
        kwave_permutation_key="perm",
    ),
    CachedParityCase(
        script="tvsp_3D_simulation_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(4096, 444),
        kwave_permutation_key="perm",
    ),
    CachedParityCase(
        script="tvsp_snells_law_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("p_final",),
        expected_shape=(128, 128),
    ),
    CachedParityCase(
        script="na_source_smoothing_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("no_window_trace", "hanning_trace", "blackman_trace"),
        expected_shape=(2131,),
        metric_method="compute_trace_metrics",
        runtime_keys=(
            "no_window_runtime_s",
            "hanning_runtime_s",
            "blackman_runtime_s",
        ),
    ),
    CachedParityCase(
        script="na_filtering_part_1_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(256, 1024),
    ),
    CachedParityCase(
        script="ivp_1D_simulation_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(2, 4267),
    ),
    CachedParityCase(
        script="ivp_binary_sensor_mask_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(178, 604),
        kwave_permutation_key="sensor_row_perm",
    ),
    CachedParityCase(
        script="ivp_homogeneous_medium_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(50, 604),
        kwave_permutation_key="sensor_row_perm",
    ),
    CachedParityCase(
        script="ivp_heterogeneous_medium_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(50, 725),
        kwave_permutation_key="sensor_row_perm",
    ),
    CachedParityCase(
        script="ivp_loading_external_image_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(50, 604),
        kwave_permutation_key="sensor_row_perm",
    ),
    CachedParityCase(
        script="na_filtering_part_2_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(256, 1024),
    ),
    CachedParityCase(
        script="na_filtering_part_3_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(256, 1024),
    ),
    CachedParityCase(
        script="na_modelling_nonlinearity_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("trace",),
        expected_shape=(1200,),
        metric_method="compute_trace_metrics",
    ),
    CachedParityCase(
        script="sd_directivity_modelling_3D_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("element_data",),
        expected_shape=(370, 11),
    ),
    CachedParityCase(
        script="tvsp_doppler_effect_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(4500,),
        metric_method="compute_trace_metrics",
    ),
    CachedParityCase(
        script="tvsp_homogeneous_medium_dipole_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("pressure",),
        expected_shape=(1, 604),
        metric_method="compute_trace_metrics",
    ),
    CachedParityCase(
        script="tvsp_homogeneous_medium_monopole_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("trace",),
        expected_shape=(604,),
        metric_method="compute_trace_metrics",
    ),
    CachedParityCase(
        script="tvsp_steering_linear_array_compare.py",
        kwave_cache_attr="_KWAVE_CACHE",
        pykwavers_cache_attr="_PKWAV_CACHE",
        arrays=("p_final",),
        expected_shape=(128, 128),
    ),
)


def _thresholds_for(module, array_name: str) -> dict[str, float]:
    thresholds = module.PARITY_THRESHOLDS
    if array_name in thresholds:
        return thresholds[array_name]
    return thresholds


def _threshold_value(thresholds: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in thresholds:
            return thresholds[name]
    return None


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.parametrize("case", CASES, ids=[case.script for case in CASES])
def test_cached_example_parity_metrics(case: CachedParityCase):
    module = load_example_module(case.script)
    assert_cached_example_artifacts(module)
    kwave_cache_path = getattr(module, case.kwave_cache_attr)
    pykwavers_cache_path = getattr(module, case.pykwavers_cache_attr)

    assert kwave_cache_path.exists()
    assert pykwavers_cache_path.exists()

    kwave = load_numeric_cache(kwave_cache_path)
    pykwavers = load_numeric_cache(pykwavers_cache_path)

    if "dt" in kwave:
        assert float(kwave["dt"]) > 0.0
    if "dt" in kwave and "dt" in pykwavers:
        assert float(pykwavers["dt"]) == float(kwave["dt"])
    if "nt" in kwave and "nt" in pykwavers:
        assert int(pykwavers["nt"]) == int(kwave["nt"])
    for runtime_key in case.runtime_keys:
        assert float(kwave[runtime_key]) > 0.0
        assert float(pykwavers[runtime_key]) > 0.0
    perm = None
    if case.kwave_permutation_key is not None:
        perm = np.asarray(
            module.build_shared_inputs()[case.kwave_permutation_key],
            dtype=np.int64,
        )
    metric_fn = getattr(module, case.metric_method)

    for array_name in case.arrays:
        kwave_array = np.asarray(kwave[array_name], dtype=np.float64)
        if perm is not None:
            kwave_array = kwave_array[perm]
        pykwavers_array = np.asarray(pykwavers[array_name], dtype=np.float64)
        thresholds = _thresholds_for(module, array_name)
        metrics = metric_fn(kwave_array, pykwavers_array)

        assert kwave_array.shape == case.expected_shape
        assert pykwavers_array.shape == case.expected_shape
        assert np.all(np.isfinite(kwave_array))
        assert np.all(np.isfinite(pykwavers_array))
        assert float(np.max(np.abs(kwave_array))) > 0.0
        assert float(np.max(np.abs(pykwavers_array))) > 0.0

        pearson_min = _threshold_value(thresholds, "pearson_r", "pearson_r_min")
        psnr_min = _threshold_value(thresholds, "psnr_db", "psnr_db_min")

        assert pearson_min is not None, (case.script, array_name)
        assert metrics["pearson_r"] >= pearson_min, (case.script, array_name)
        assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"], (
            case.script,
            array_name,
        )
        assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"], (
            case.script,
            array_name,
        )
        if psnr_min is not None:
            assert metrics["psnr_db"] >= psnr_min, (
                case.script,
                array_name,
            )


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_cached_directional_array_element_parity_metrics():
    module = load_example_module("sd_directional_array_elements_compare.py")
    assert_cached_example_artifacts(module)

    if not module._KWAVE_CACHE.exists() or not module._PKWAV_CACHE.exists():
        inputs = module.build_shared_inputs()
        module.run_kwave(inputs, no_cache=True)
        module.run_pykwavers(inputs, no_cache=True)

    assert module._KWAVE_CACHE.exists()
    assert module._PKWAV_CACHE.exists()

    kwave = load_numeric_cache(module._KWAVE_CACHE)
    pykwavers = load_numeric_cache(module._PKWAV_CACHE)

    assert int(kwave["cache_version"]) == module.CACHE_VERSION
    assert int(pykwavers["cache_version"]) == module.CACHE_VERSION
    assert int(pykwavers["nt"]) == int(kwave["nt"])
    assert float(pykwavers["dt"]) == float(kwave["dt"])
    assert float(kwave["runtime_s"]) > 0.0
    assert float(pykwavers["runtime_s"]) > 0.0

    inputs = module.build_shared_inputs()
    nt = int(kwave["nt"])
    kwave_element_data = module.compute_element_data(
        np.asarray(kwave["p_raw"], dtype=np.float64),
        inputs["element_kw_rows"],
        nt,
    )
    pykwavers_element_data = module.compute_element_data(
        np.asarray(pykwavers["p_raw"], dtype=np.float64),
        inputs["element_py_rows"],
        nt,
    )
    metrics = module.compute_image_metrics(kwave_element_data, pykwavers_element_data)
    thresholds = module.PARITY_THRESHOLDS

    assert kwave_element_data.shape == (module.NE, nt)
    assert pykwavers_element_data.shape == kwave_element_data.shape
    assert np.all(np.isfinite(kwave_element_data))
    assert np.all(np.isfinite(pykwavers_element_data))
    assert float(np.max(np.abs(kwave_element_data))) > 0.0
    assert float(np.max(np.abs(pykwavers_element_data))) > 0.0
    assert metrics["pearson_r"] >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= metrics["rms_ratio"]
    assert metrics["rms_ratio"] <= thresholds["rms_ratio_max"]
    assert metrics["psnr_db"] >= thresholds["psnr_db"]


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_cached_particle_velocity_dominant_axis_parity():
    module = load_example_module("ivp_recording_particle_velocity_compare.py")

    if not module._KWAVE_CACHE.exists() or not module._PKWAV_CACHE.exists():
        module.run_kwave_python()
        module.run_pykwavers()

    for path in (
        module.REPORT_PATH,
        module.PRESSURE_FIGURE_PATH,
        module.UX_FIGURE_PATH,
        module.UY_FIGURE_PATH,
        module._KWAVE_CACHE,
        module._PKWAV_CACHE,
    ):
        assert path.exists(), path.name

    metrics_text = module.REPORT_PATH.read_text(encoding="utf-8", errors="ignore")
    assert "parity_status: PASS" in metrics_text

    kwave = load_numeric_cache(module._KWAVE_CACHE)
    pykwavers = load_numeric_cache(module._PKWAV_CACHE)

    assert int(kwave["cache_version"]) == module.CACHE_VERSION
    assert int(pykwavers["cache_version"]) == module.CACHE_VERSION
    assert float(kwave["runtime_s"]) > 0.0
    assert float(pykwavers["runtime_s"]) > 0.0
    assert float(kwave["dt"]) > 0.0
    assert float(pykwavers["dt"]) > 0.0

    nt = min(int(kwave["nt"]), int(pykwavers["nt"]))
    reorder = np.asarray(module.SENSOR_REORDER, dtype=np.int64)
    thresholds = module.PARITY_THRESHOLDS

    kwave_p = np.asarray(kwave["p"], dtype=np.float64)[:, :nt]
    kwave_ux = np.asarray(kwave["ux"], dtype=np.float64)[:, :nt]
    kwave_uy = np.asarray(kwave["uy"], dtype=np.float64)[:, :nt]
    pykwavers_p = np.asarray(pykwavers["p"], dtype=np.float64)[:, :nt][reorder]
    pykwavers_ux = np.asarray(pykwavers["ux"], dtype=np.float64)[:, :nt][reorder]
    pykwavers_uy = np.asarray(pykwavers["uy"], dtype=np.float64)[:, :nt][reorder]

    for array in (
        kwave_p,
        kwave_ux,
        kwave_uy,
        pykwavers_p,
        pykwavers_ux,
        pykwavers_uy,
    ):
        assert array.shape == (4, nt)
        assert np.all(np.isfinite(array))
        assert float(np.max(np.abs(array))) > 0.0

    for index, label in enumerate(module.SENSOR_LABELS):
        pearson = module.pearson_r(kwave_p[index], pykwavers_p[index])
        assert pearson >= thresholds["pressure_pearson_r"], label

    for index, (label, expect_ux) in enumerate(
        zip(module.SENSOR_LABELS, module.SENSOR_EXPECT_UX)
    ):
        rms_ux = float(np.sqrt(np.mean(pykwavers_ux[index] ** 2)))
        rms_uy = float(np.sqrt(np.mean(pykwavers_uy[index] ** 2)))
        assert (rms_ux > rms_uy) == expect_ux, label

        if expect_ux:
            pearson = module.pearson_r(kwave_ux[index], pykwavers_ux[index])
        else:
            pearson = module.pearson_r(kwave_uy[index], pykwavers_uy[index])
        assert pearson >= thresholds["dominant_velocity_pearson_r"], label


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
def test_cached_tiny_phased_array_scan_line_summary():
    module = load_example_module("us_bmode_phased_array_tiny_compare.py")
    assert_cached_example_artifacts(module)

    assert module.KWAVE_CACHE.exists()
    assert module.PYKWAVERS_CACHE.exists()

    kwave = load_numeric_cache(module.KWAVE_CACHE)
    pykwavers = load_numeric_cache(module.PYKWAVERS_CACHE)

    assert int(kwave["cache_version"]) == module.CACHE_VERSION
    assert int(pykwavers["cache_version"]) == module.CACHE_VERSION
    assert int(pykwavers["nt"]) == int(kwave["nt"])
    assert int(pykwavers["seed"]) == int(kwave["seed"])
    assert np.array_equal(kwave["steering_angles"], pykwavers["steering_angles"])
    assert float(kwave["runtime"]) > 0.0
    assert float(pykwavers["runtime"]) > 0.0

    kwave_lines = np.asarray(kwave["scan_lines"], dtype=np.float64)
    pykwavers_lines = np.asarray(pykwavers["scan_lines"], dtype=np.float64)
    assert kwave_lines.shape == (5, int(kwave["nt"]))
    assert pykwavers_lines.shape == kwave_lines.shape
    assert np.all(np.isfinite(kwave_lines))
    assert np.all(np.isfinite(pykwavers_lines))
    assert float(np.max(np.abs(kwave_lines))) > 0.0
    assert float(np.max(np.abs(pykwavers_lines))) > 0.0

    trace_metrics = [
        module.compute_trace_metrics(kwave_lines[index], pykwavers_lines[index])
        for index in range(kwave_lines.shape[0])
    ]
    mean_rms_ratio = float(np.mean([metrics["rms_ratio"] for metrics in trace_metrics]))
    mean_pearson = float(np.mean([metrics["pearson_r"] for metrics in trace_metrics]))
    image_metrics = module.compute_image_metrics(kwave_lines, pykwavers_lines)
    thresholds = module.PARITY_THRESHOLDS

    assert mean_pearson >= thresholds["mean_pearson_r"]
    assert thresholds["mean_rms_ratio_min"] <= mean_rms_ratio
    assert mean_rms_ratio <= thresholds["mean_rms_ratio_max"]
    assert image_metrics["rms_ratio"] >= thresholds["mean_rms_ratio_min"]
    assert image_metrics["rms_ratio"] <= thresholds["mean_rms_ratio_max"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
