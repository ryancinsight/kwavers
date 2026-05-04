import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / "example_parity_utils.py"
    spec = importlib.util.spec_from_file_location("example_parity_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_example_module(name: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_trace_metrics_identity():
    module = _load_module()
    x = np.linspace(-1.0, 1.0, 128)
    metrics = module.compute_trace_metrics(x, x.copy())
    assert abs(metrics["pearson_r"] - 1.0) < 1e-12
    assert abs(metrics["rms_ratio"] - 1.0) < 1e-12
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs_diff"] == 0.0


def test_image_metrics_identity():
    module = _load_module()
    img = np.arange(64, dtype=float).reshape(8, 8)
    metrics = module.compute_image_metrics(img, img.copy())
    assert abs(metrics["pearson_r"] - 1.0) < 1e-12
    assert abs(metrics["rms_ratio"] - 1.0) < 1e-12
    assert metrics["psnr_db"] > 300.0


def test_sensor_matrix_summary_identity():
    module = _load_module()
    mat = np.arange(24, dtype=float).reshape(4, 6)
    summary = module.summarize_sensor_matrix_metrics(mat, mat.copy(), expected_sensors=4)
    assert summary["n_sensors"] == 4.0
    assert summary["n_time_samples"] == 6.0
    assert abs(summary["pearson_r_mean"] - 1.0) < 1e-12
    assert abs(summary["pearson_r_median"] - 1.0) < 1e-12
    assert abs(summary["rms_ratio_mean"] - 1.0) < 1e-12
    assert abs(summary["rms_ratio_median"] - 1.0) < 1e-12
    assert summary["rmse_median"] == 0.0
    assert summary["max_abs_diff_max"] == 0.0


def test_sensor_matrix_summary_transpose_alignment():
    module = _load_module()
    mat = np.arange(24, dtype=float).reshape(4, 6)
    summary = module.summarize_sensor_matrix_metrics(mat.T, mat, expected_sensors=4)
    assert summary["n_sensors"] == 4.0
    assert summary["n_time_samples"] == 6.0
    assert abs(summary["pearson_r_mean"] - 1.0) < 1e-12


def test_pml_outside_padding_shapes_and_values():
    module = _load_module()

    volume_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    padded_2d = module.pad_volume_for_pml_outside(volume_2d, (1, 2, 3))
    assert padded_2d.shape == (4, 6, 1)
    assert padded_2d[1:3, 2:4, 0].tolist() == volume_2d.tolist()
    assert np.count_nonzero(padded_2d) == volume_2d.size

    volume_3d = np.ones((2, 3, 4), dtype=float)
    padded_3d = module.pad_volume_for_pml_outside(volume_3d, (2, 1, 1))
    assert padded_3d.shape == (6, 5, 6)
    assert np.allclose(padded_3d[2:4, 1:4, 1:5], volume_3d)

    assert module.expand_pml_outside_shape((2, 3), (1, 2, 3)) == (4, 7, 1)
    assert module.expand_pml_outside_shape((2, 3, 4), (1, 2, 3)) == (4, 7, 10)


def test_pml_outside_padding_rejects_2d_tuple_for_3d_volume():
    module = _load_module()

    volume_3d = np.ones((2, 3, 4), dtype=float)
    try:
        module.pad_volume_for_pml_outside(volume_3d, (1, 2))
    except ValueError as exc:
        assert "3-D PML tuple" in str(exc)
    else:
        raise AssertionError("Expected pad_volume_for_pml_outside to reject a 2-D PML tuple")


def test_running_timing_stats_accumulates_summary_without_per_line_materialization():
    module = _load_module()
    stats = module.RunningTimingStats(("a_ns", "b_ns", "c_ns"))
    stats.update((1_000_000, 2_000_000, 3_000_000))
    stats.update((2_000_000, 4_000_000, 6_000_000))

    summary = stats.summary_ms()
    lines = stats.format_lines()
    assert stats.count == 2
    assert len(lines) == 3
    assert lines[0].startswith("  a_ns")
    assert "mean=1.500" in lines[0]
    assert summary["a_ns"]["mean"] == 1.5
    assert summary["a_ns"]["min"] == 1.0
    assert summary["a_ns"]["max"] == 2.0
    assert summary["b_ns"]["mean"] == 3.0
    assert summary["c_ns"]["max"] == 6.0


def test_advance_lateral_window_inplace_matches_full_slice_copy():
    module = _load_module()
    source = np.arange(3 * 8 * 2, dtype=float).reshape(3, 8, 2)
    current = source[:, 1:5, :].copy()
    module.advance_lateral_window_inplace(current, source, 1, 3)

    expected = source[:, 3:7, :]
    assert np.array_equal(current, expected)


def test_side_by_side_parity_figure_exports_reference_candidate_and_difference(tmp_path):
    module = _load_module()
    reference = np.zeros((4, 5, 6), dtype=float)
    candidate = np.zeros_like(reference)
    reference[2, 1:4, 2:5] = 1.0
    candidate[2, 2:5, 2:5] = 1.0

    path = module.save_side_by_side_parity_figure(
        reference,
        candidate,
        tmp_path / "visual_parity.png",
        title="unit visual parity",
        projection="peak_slice",
        axis=0,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_side_by_side_parity_figure_rejects_shape_mismatch(tmp_path):
    module = _load_module()
    try:
        module.save_side_by_side_parity_figure(
            np.zeros((2, 3)),
            np.zeros((2, 4)),
            tmp_path / "bad.png",
            title="bad visual parity",
        )
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected visual parity export to reject mismatched shapes")


def test_comparative_examples_declare_visual_exports():
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    visual_markers = (
        "save_side_by_side_parity_figure",
        "save_comparison_figure",
        "savefig",
        "plot_comparison",
        "write_plots",
        "FIGURE_PATH",
        "PNG_PATH",
    )
    missing = []
    for path in sorted(examples_dir.glob("*.py")):
        if not any(token in path.name for token in ("compare", "comparison", "parity")):
            continue
        text = path.read_text(encoding="utf-8")
        if not any(marker in text for marker in visual_markers):
            missing.append(path.name)

    assert missing == []


def test_ivp_particle_velocity_uses_shared_smoothed_source_boundary():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "ivp_recording_particle_velocity_compare.py"
    text = script.read_text(encoding="utf-8")

    assert "kwave_smooth" in text
    assert "make_disc_p0_2d_raw" in text
    assert "smooth_p0=False" in text
    assert "source_preprocessing:" in text


def test_hifu_procedure_focus_and_temperature_are_value_semantic():
    module = _load_example_module("hifu_procedure_simulation")
    acoustic = module.AcousticConfig(
        target_peak_intensity_w_m2=4.0e5,
        aperture_radial_samples=12,
        aperture_angular_samples=32,
    )
    thermal = module.ThermalConfig(sonication_s=2.0, cooling_s=2.0, dt_s=0.1)
    grid_config = module.GridConfig(x_extent_m=8.0e-3, z_min_m=20.0e-3, z_max_m=45.0e-3, dx_m=0.5e-3, dz_m=1.0e-3)

    x, z, xx, zz = module.build_grid(grid_config)
    intensity = module.focused_aperture_intensity(xx, zz, acoustic)
    metrics = module.focal_metrics(intensity, x, z)
    pressure_peak_mpa, mechanical_index, cavitation = module.cavitation_metrics(intensity, acoustic, grid_config)
    heat_source = 2.0 * acoustic.absorption_np_m * intensity
    times, focus_temperature, max_temperature, final_temperature = module.pennes_temperature(
        heat_source,
        grid_config,
        thermal,
    )

    assert abs(metrics["focus_x_m"]) <= grid_config.dx_m
    assert abs(metrics["focus_z_m"] - acoustic.focal_length_m) <= 4.0 * grid_config.dz_m
    assert metrics["peak_intensity_w_m2"] == np.max(intensity)
    assert metrics["lateral_fwhm_m"] > 0.0
    assert metrics["axial_fwhm_m"] > 0.0
    expected_pressure_mpa = np.sqrt(2.0 * acoustic.density_kg_m3 * acoustic.sound_speed_m_s * intensity) / 1.0e6
    assert np.allclose(pressure_peak_mpa, expected_pressure_mpa)
    assert np.allclose(mechanical_index, pressure_peak_mpa / np.sqrt(acoustic.frequency_hz / 1.0e6))
    assert cavitation["peak_pressure_mpa"] == np.max(pressure_peak_mpa)
    assert cavitation["peak_mechanical_index"] == np.max(mechanical_index)
    assert cavitation["mi_threshold_area_mm2"] >= 0.0
    assert focus_temperature[int(round(thermal.sonication_s / thermal.dt_s))] > thermal.baseline_temperature_c
    assert focus_temperature[-1] < np.max(focus_temperature)
    assert np.max(max_temperature) >= np.max(focus_temperature)
    assert np.max(final_temperature) >= focus_temperature[-1]


def test_hifu_cavitation_feedback_uses_bubble_receiver_signal():
    module = _load_example_module("hifu_procedure_simulation")
    acoustic = module.AcousticConfig(target_peak_intensity_w_m2=8.0e5)
    thermal = module.ThermalConfig(sonication_s=1.0, cooling_s=0.5, dt_s=0.1)
    grid_config = module.GridConfig(x_extent_m=6.0e-3, z_min_m=24.0e-3, z_max_m=40.0e-3, dx_m=1.0e-3, dz_m=1.0e-3)
    bubble = module.BubbleConfig(dt_s=5.0e-9, cycles_per_window=4, control_windows=12)

    feedback = module.simulate_cavitation_feedback(acoustic, bubble, peak_pressure_pa=1.6e6)
    radius = np.asarray(feedback["radius_m"], dtype=float)
    receiver = np.asarray(feedback["receiver_pressure_pa"], dtype=float)
    activity = np.asarray(feedback["receiver_activity"], dtype=float)
    radius_activity = np.asarray(feedback["radius_activity"], dtype=float)
    controller = np.asarray(feedback["controller_output"], dtype=float)
    times = np.linspace(0.0, thermal.sonication_s + thermal.cooling_s, int(round((thermal.sonication_s + thermal.cooling_s) / thermal.dt_s)) + 1)
    power_envelope = module.feedback_power_envelope(times, thermal, feedback, bubble.nominal_pressure_fraction)
    x, z, xx, zz = module.build_grid(grid_config)
    intensity = module.focused_aperture_intensity(xx, zz, acoustic)
    heat_source = 2.0 * acoustic.absorption_np_m * intensity
    _, uncontrolled_focus_temperature, _, _ = module.pennes_temperature(heat_source, grid_config, thermal)
    _, controlled_focus_temperature, _, _ = module.pennes_temperature(heat_source, grid_config, thermal, power_envelope)

    assert np.all(np.isfinite(radius))
    assert np.all(np.isfinite(receiver))
    assert np.all(np.isfinite(activity))
    assert np.all(np.isfinite(radius_activity))
    assert np.all(controller >= bubble.min_pressure_fraction)
    assert np.all(controller <= bubble.max_pressure_fraction)
    assert np.max(radius) > bubble.equilibrium_radius_m
    assert np.max(np.abs(receiver)) > 0.0
    assert np.max(controller) > controller[0]
    assert feedback["max_radius_ratio"] == np.max(radius) / bubble.equilibrium_radius_m
    assert np.max(radius) / bubble.equilibrium_radius_m <= bubble.target_inertial_radius_ratio * 1.1
    assert np.all(power_envelope[times > thermal.sonication_s] == 0.0)
    assert np.max(power_envelope[times <= thermal.sonication_s]) <= (bubble.max_pressure_fraction / bubble.nominal_pressure_fraction) ** 2
    assert power_envelope[int(round(thermal.sonication_s / thermal.dt_s))] == (feedback["final_pressure_fraction"] / bubble.nominal_pressure_fraction) ** 2
    assert np.max(controlled_focus_temperature) < np.max(uncontrolled_focus_temperature)
    assert controlled_focus_temperature[-1] < uncontrolled_focus_temperature[-1]
    assert np.max(controlled_focus_temperature) - thermal.baseline_temperature_c > 0.25 * (
        np.max(uncontrolled_focus_temperature) - thermal.baseline_temperature_c
    )
