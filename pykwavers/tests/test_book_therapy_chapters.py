from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np

BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_manifest_includes_bbb_and_hifu_chapters_with_scripts():
    manifest = tomllib.loads((BOOK_DIR / "chapters.toml").read_text(encoding="utf-8"))
    chapters = {int(entry["number"]): entry for entry in manifest["chapter"]}

    assert chapters[24]["script"] == "ch24_bbb_lifu_opening.py"
    assert "BBB" in chapters[24]["title"]
    assert (BOOK_DIR / chapters[24]["script"]).is_file()

    assert chapters[25]["script"] == "ch25_transcranial_brain_fus_planning.py"
    assert "HIFU" in chapters[25]["title"]
    assert (BOOK_DIR / chapters[25]["script"]).is_file()

    assert chapters[26]["script"] == "ch26_neuromodulation.py"
    assert "Neuromodulation" in chapters[26]["title"]
    assert (BOOK_DIR / chapters[26]["script"]).is_file()

    assert chapters[27]["script"] == "ch27_transcranial_ust_brain_imaging.py"
    assert "Seismic" in chapters[27]["title"]
    assert (BOOK_DIR / chapters[27]["script"]).is_file()

    assert chapters[28]["script"] == "ch28_abdominal_histotripsy_fwi.py"
    assert "Abdominal" in chapters[28]["title"]
    assert (BOOK_DIR / chapters[28]["script"]).is_file()

    assert chapters[29]["script"] == "ch29_theranostic_fwi_platforms.py"
    assert "Same-Device" in chapters[29]["title"]
    assert (BOOK_DIR / chapters[29]["script"]).is_file()

    assert chapters[30]["script"] == "ch30_intravascular_ultrasound.py"
    assert "Intravascular" in chapters[30]["title"]
    assert (BOOK_DIR / chapters[30]["script"]).is_file()

    assert chapters[32]["script"] == "ch32_segmented_tissue_transducer_optimization.py"
    assert "Segmented Tissue" in chapters[32]["title"]
    assert (BOOK_DIR / chapters[32]["script"]).is_file()


def test_book_readme_links_bbb_and_hifu_markdown_chapters():
    readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")

    assert "(bbb_lifu_opening.md)" in readme
    assert "(hifu_transcranial_ablation.md)" in readme
    assert "(neuromodulation.md)" in readme
    assert "(transcranial_ust_brain_imaging.md)" in readme
    assert "(abdominal_histotripsy_fwi.md)" in readme
    assert "(theranostic_fwi_platforms.md)" in readme
    assert "(intravascular_ultrasound.md)" in readme
    assert "(segmented_tissue_transducer_planning.md)" in readme
    assert (DOCS_DIR / "bbb_lifu_opening.md").is_file()
    assert (DOCS_DIR / "hifu_transcranial_ablation.md").is_file()
    assert (DOCS_DIR / "neuromodulation.md").is_file()
    assert (DOCS_DIR / "transcranial_ust_brain_imaging.md").is_file()
    assert (DOCS_DIR / "abdominal_histotripsy_fwi.md").is_file()
    assert (DOCS_DIR / "theranostic_fwi_platforms.md").is_file()
    assert (DOCS_DIR / "intravascular_ultrasound.md").is_file()
    assert (DOCS_DIR / "segmented_tissue_transducer_planning.md").is_file()


def test_chapter29_layout_helpers_report_skin_and_focused_bowl_clearance():
    import ch29_theranostic_fwi_platforms as ch29

    kidney = {
        "anatomy": "kidney",
        "placement_metrics": {
            "skin_contact_to_nearest_aperture_m": 0.003,
            "min_body_clearance_m": 0.003,
        },
    }
    brain = {
        "anatomy": "brain",
        "placement_metrics": {
            "skin_contact_to_nearest_aperture_m": 0.016,
            "min_body_clearance_m": 0.015,
        },
    }

    x_limits = ch29.axis_limits(
        [-0.04, 0.04],
        np.asarray([-0.11, 0.09]),
        np.asarray([-0.043]),
    )
    y_limits = ch29.axis_limits([-0.05, 0.05], np.asarray([]), np.asarray([]))

    assert "skin gap 3.0 mm" == ch29.placement_label(kidney)
    assert "focused-bowl clearance 15.0 mm" == ch29.placement_label(brain)
    assert x_limits[0] < -0.11
    assert x_limits[1] > 0.09
    assert y_limits[0] < -0.05
    assert y_limits[1] > 0.05


def test_chapter29_fig02_reconstruction_grid_starts_with_ct_context():
    import ch29_theranostic_fwi_platforms as ch29

    first_key, first_cmap, first_title = ch29.RECONSTRUCTION_FIGURE_COLUMNS[0]
    second_key = ch29.RECONSTRUCTION_FIGURE_COLUMNS[1][0]
    rendered_keys = [key for key, _, _ in ch29.RECONSTRUCTION_FIGURE_COLUMNS]
    reconstruction_keys = [key for key, _ in ch29.RECONSTRUCTION_CHANNELS]

    assert first_key == "ct_hu"
    assert first_cmap == "gray"
    assert first_title == "CT + target + tx/rx"
    assert second_key == "exposure"
    assert rendered_keys[-len(reconstruction_keys) :] == reconstruction_keys
    assert "elastic_shear_reconstruction" in reconstruction_keys
    assert all("FWI" not in title for _, title in ch29.RECONSTRUCTION_CHANNELS)
    assert all("FWI" not in title for _, _, title in ch29.RECONSTRUCTION_FIGURE_COLUMNS)


def test_chapter29_controlled_comparison_grid_starts_with_ct_context():
    import ch29_theranostic_fwi_platforms as ch29

    first_key, first_cmap, first_title = ch29.CONTROLLED_COMPARISON_COLUMNS[0]
    second_key = ch29.CONTROLLED_COMPARISON_COLUMNS[1][0]

    assert first_key == "placement_ct_hu"
    assert first_cmap == "gray"
    assert first_title == "CT + target + tx/rx"
    assert second_key == "common_target"
    assert any(key == "elastic_shear" for key, _, _ in ch29.CONTROLLED_COMPARISON_COLUMNS)


def test_chapter29_fig05_defaults_to_fig02_case_grids(monkeypatch):
    import ch29_theranostic_fwi_platforms as ch29

    for key in (
        "KWAVERS_CH29_NONLINEAR_GRID",
        "KWAVERS_CH29_BRAIN_NONLINEAR_GRID",
        "KWAVERS_CH29_KIDNEY_NONLINEAR_GRID",
        "KWAVERS_CH29_LIVER_NONLINEAR_GRID",
    ):
        monkeypatch.delenv(key, raising=False)

    default_grids = {case["name"]: ch29.nonlinear_grid_size(case) for case in ch29.CASES}

    assert default_grids == {"brain": 56, "kidney": 56, "liver": 56}

    monkeypatch.setenv("KWAVERS_CH29_NONLINEAR_GRID", "40")
    assert {case["name"]: ch29.nonlinear_grid_size(case) for case in ch29.CASES} == {
        "brain": 40,
        "kidney": 40,
        "liver": 40,
    }

    monkeypatch.setenv("KWAVERS_CH29_KIDNEY_NONLINEAR_GRID", "56")
    kidney = next(case for case in ch29.CASES if case["name"] == "kidney")

    assert ch29.nonlinear_grid_size(kidney) == 56


def test_chapter29_output_directory_can_be_overridden(monkeypatch):
    monkeypatch.setenv("KWAVERS_CH29_OUT_DIR", "D:/kwavers/target/ch29-test-output")

    import importlib
    import ch29_theranostic_fwi_platforms as ch29

    reloaded = importlib.reload(ch29)

    assert reloaded.OUT_DIR.as_posix().endswith("target/ch29-test-output")


def test_chapter29_loader_rejects_stale_nonlinear_extension_signature():
    import ch29_theranostic_fwi_platforms as ch29

    def inverse_stub():
        return None

    def current_nonlinear_stub():
        return None

    def stale_nonlinear_stub():
        return None

    class CurrentExtension:
        run_theranostic_inverse_from_ritk = staticmethod(inverse_stub)
        run_theranostic_nonlinear_3d_from_ritk = staticmethod(current_nonlinear_stub)

    class StaleExtension:
        run_theranostic_inverse_from_ritk = staticmethod(inverse_stub)
        run_theranostic_nonlinear_3d_from_ritk = staticmethod(stale_nonlinear_stub)

    current_nonlinear_stub.__text_signature__ = (
        "(ct_nifti_path, treatment_window_radius_m=0.04, min_points_per_wavelength=6.0)"
    )
    stale_nonlinear_stub.__text_signature__ = (
        "(ct_nifti_path, lesion_delta_c_m_s=-35.0)"
    )

    assert ch29.pykwavers_extension_is_current(CurrentExtension)
    assert not ch29.pykwavers_extension_is_current(StaleExtension)


def test_chapter29_nonlinear_defaults_to_histotripsy_drive(monkeypatch):
    import ch29_theranostic_fwi_platforms as ch29

    monkeypatch.delenv("KWAVERS_CH29_NONLINEAR_SOURCE_PRESSURE_PA", raising=False)
    monkeypatch.delenv("KWAVERS_CH29_NONLINEAR_FREQUENCY_HZ", raising=False)

    frequencies = {case["name"]: ch29.nonlinear_frequency_hz(case) for case in ch29.CASES}

    assert frequencies == {"brain": 650_000.0, "kidney": 500_000.0, "liver": 500_000.0}

    for case in ch29.CASES:
        pressure = ch29.nonlinear_source_pressure_pa(case)
        frequency_mhz = ch29.nonlinear_frequency_hz(case) * 1.0e-6
        mi = pressure * 1.0e-6 / np.sqrt(frequency_mhz)

        assert pressure >= float(case["pressure"])
        assert mi >= ch29.INERTIAL_MI_THRESHOLD

    monkeypatch.setenv("KWAVERS_CH29_NONLINEAR_SOURCE_PRESSURE_PA", "3.0e6")
    assert ch29.nonlinear_source_pressure_pa(ch29.CASES[0]) == 3.0e6


def test_chapter29_ct_context_draws_transducer_locations():
    import matplotlib.pyplot as plt

    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((5, 5), dtype=bool)
    target[2, 2] = True
    result = {
        "anatomy": "kidney",
        "device_model": "focused_bowl_256_element_skin_coupled_arc",
        "element_count": 2,
        "placement_ct_hu": np.zeros((5, 5), dtype=float),
        "placement_spacing_m": (0.01, 0.01),
        "placement_target_mask": target,
        "placement_body_mask": np.ones((5, 5), dtype=bool),
        "placement_therapy_points_m": np.asarray([[-0.03, 0.02], [0.03, 0.02]], dtype=float),
        "placement_imaging_points_m": np.asarray([[0.0, 0.035]], dtype=float),
        "placement_focus_m": [0.0, 0.0],
        "placement_skin_contact_m": [0.0, 0.02],
        "placement_context_skin_gap_m": 0.003,
        "placement_metrics": {"min_body_clearance_m": 0.003},
    }

    fig, ax = plt.subplots()
    try:
        ch29.plot_placement_ct(ax, result, show_legend=False)
        scatter_offsets = [
            np.asarray(collection.get_offsets(), dtype=float)
            for collection in ax.collections
            if hasattr(collection, "get_offsets") and len(collection.get_offsets()) > 0
        ]
        offsets = np.vstack(scatter_offsets)

        assert np.any(np.all(np.isclose(offsets, [-0.03, 0.02]), axis=1))
        assert np.any(np.all(np.isclose(offsets, [0.03, 0.02]), axis=1))
        assert np.any(np.all(np.isclose(offsets, [0.0, 0.035]), axis=1))
        assert ax.get_xlim()[0] < -0.03
        assert ax.get_xlim()[1] > 0.03
    finally:
        plt.close(fig)


def test_active_book_focused_bowl_artifacts_use_generic_source_labels():
    checked_paths = [
        BOOK_DIR / "ch31_clinical_device_geometry.py",
        DOCS_DIR / "clinical_device_geometry.md",
        DOCS_DIR / "figures" / "ch31" / "metrics.json",
        DOCS_DIR / "figures" / "ch29" / "metrics.json",
    ]
    forbidden_source_identity_tokens = (
        "HistoSonics",
        "InSightec",
        "Exablate",
        "histosonics_like",
        "insightec_like",
        "brain_helmet",
        "helmet",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_source_identity_tokens:
            assert token not in text, (path, token)

    ch31_metrics = json.loads((DOCS_DIR / "figures" / "ch31" / "metrics.json").read_text(encoding="utf-8"))
    assert ch31_metrics["inverse_results"]["brain"]["device_model"] == "transcranial_focused_bowl_projection"
    assert (
        ch31_metrics["inverse_results"]["liver"]["device_model"]
        == "focused_bowl_256_element_skin_coupled_arc"
    )
    assert (
        ch31_metrics["inverse_results"]["kidney"]["device_model"]
        == "focused_bowl_256_element_skin_coupled_arc"
    )


def test_chapter29_reconstruction_diagnostics_quantify_outside_target_sidelobes():
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((4, 4), dtype=bool)
    target[1, 1] = True
    body = np.ones((4, 4), dtype=bool)
    image = np.zeros((4, 4), dtype=float)
    image[1, 1] = 1.0
    image[2, 2] = 0.1
    result = {
        "target_mask": target,
        "body_mask": body,
        **{key: image for key, _ in ch29.RECONSTRUCTION_CHANNELS},
    }

    diagnostics = ch29.reconstruction_diagnostics(result)
    active = diagnostics["active_lesion_reconstruction"]

    assert np.isclose(ch29.normalized_db(image)[2, 2], -20.0)
    assert np.isclose(active["outside_peak_ratio"], 0.1)
    assert np.isclose(active["outside_peak_db"], -20.0)
    assert np.isclose(active["outside_energy_fraction"], 0.01 / 1.01)


def test_chapter29_controlled_comparison_uses_common_target_and_records_histories():
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((4, 4, 3), dtype=bool)
    target[1, 1, 1] = True
    target[1, 2, 1] = True
    linear_target = np.max(target, axis=2).astype(float)
    nonlinear_fusion = np.zeros_like(target, dtype=float)
    nonlinear_fusion[1, 1, 1] = 1.0
    nonlinear_fusion[2, 2, 1] = 0.5
    pressure = target.astype(float)
    pressure[3, 3, 1] = 5.0
    placement_target = np.zeros((6, 6), dtype=bool)
    placement_target[2, 2] = True
    placement_target[2, 3] = True
    linear = {
        "anatomy": "brain",
        "fused_reconstruction": linear_target,
        "elastic_shear_reconstruction": linear_target,
        "active_lesion_reconstruction": linear_target,
        "exposure": linear_target,
        "spacing_m": 1.0,
        "focus_m": (0.0, 0.0),
        "therapy_x_m": np.asarray([-1.0, 1.0]),
        "therapy_y_m": np.asarray([0.0, 0.0]),
        "placement_ct_hu": np.zeros((6, 6), dtype=float),
        "placement_spacing_m": [1.0, 1.0],
        "placement_target_mask": placement_target,
        "placement_body_mask": np.ones((6, 6), dtype=bool),
        "placement_therapy_points_m": np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=float),
        "placement_imaging_points_m": np.asarray([[0.0, 1.0]], dtype=float),
        "placement_focus_m": [0.0, 0.0],
        "placement_skin_contact_m": [0.0, 1.0],
    }
    nonlinear = {
        "anatomy": "brain",
        "target_mask": target,
        "inversion_mask": target,
        "crop_bounds_index": [0, 3, 0, 3, 0, 2],
        "source_dimensions": [4, 4, 3],
        "source_spacing_m": [1.0, 1.0, 1.0],
        "body_mask": np.ones_like(target, dtype=bool),
        "westervelt_peak_pressure_pa": pressure,
        "frequency_hz": 1.0e6,
        "source_pressure_pa": 1.0,
        "source_scale": 1.0,
        "inertial_mi_threshold": 1.9,
        "multiparameter_fwi_score": nonlinear_fusion,
        "cavitation_source_density": nonlinear_fusion,
        "reconstructed_cavitation_density": nonlinear_fusion,
        "nonlinear_fusion_score": nonlinear_fusion,
        "therapy_points_m": np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "electronic_steering_metrics": {
            "nominal_focus_index": [1, 1, 1],
            "calibration_hotspot_index": [1, 2, 1],
            "steering_focus_index": [1, 1, 1],
            "correction_grid_cells": [0, 0, 0],
            "calibration_hotspot_distance_grid_cells": 1.0,
            "steering_applied": False,
        },
        "fwi_objective_history": np.asarray([2.0, 1.25]),
        "cavitation_objective_history": np.asarray([1.0, 0.8]),
    }

    comparison = ch29.build_controlled_comparison([linear], [nonlinear])[0]

    assert comparison["common_grid_shape"] == [6, 6]
    assert comparison["fields"]["placement_ct_hu"].shape == (6, 6)
    assert np.array_equal(comparison["fields"]["common_target"], placement_target)
    assert comparison["fields"]["linear_fusion"].shape == (6, 6)
    assert comparison["fields"]["elastic_shear"].shape == (6, 6)
    assert comparison["fields"]["nonlinear_fusion"].shape == (6, 6)
    assert comparison["fields"]["nonlinear_pressure_window"].shape == (6, 6)
    assert comparison["fields"]["nonlinear_pressure_raw"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_linear_fusion"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_elastic_shear"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_nonlinear_fusion"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_common_target"].dtype == np.bool_
    assert comparison["geometry"]["comparison_frame"] == "full_ct_placement_xy_projection"
    assert comparison["geometry"]["common_target_voxels"] == 2
    assert comparison["comparison_metrics"]["linear_fusion"]["dice_equal_area"] == 1.0
    assert comparison["comparison_metrics"]["elastic_shear"]["dice_equal_area"] == 1.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["target_peak"] == 1.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["outside_peak"] == 0.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["window_ct_field_target_peak"] == 1.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["raw_peak_pressure_pa"] == 5.0
    assert (
        comparison["comparison_metrics"]["nonlinear_pressure"]["raw_ct_field_hotspot_distance_to_target_grid_cells"]
        > 0.0
    )
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["ct_frame_pressure_hotspot_distance_m"] >= 0.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["ct_frame_pressure_hotspot_cross_axis_offset_m"] >= 0.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["electronic_steering_steering_applied"] is False
    assert comparison["geometry"]["planned_to_nonlinear_aperture_axis_angle_deg"] >= 0.0
    assert comparison["comparison_metrics"]["nonlinear_cavitation_source"]["target_peak"] == 1.0
    assert comparison["objective_history"]["nonlinear_fwi"] == [2.0, 1.25]
    assert "linear fusion Dice" in comparison["technical_explanation"]
    assert "pressure hotspot offset" in comparison["technical_explanation"]


def test_chapter29_controlled_comparison_linear_extent_uses_crop_metadata():
    import ch29_controlled_comparison as comparison

    result = {
        "fused_reconstruction": np.zeros((3, 3), dtype=float),
        "spacing_m": 1.0,
        "source_dimensions": [7, 5],
        "source_spacing_m": [0.002, 0.003],
        "crop_bounds_index": [2, 4, 1, 3],
    }

    assert comparison._linear_extent(result) == [-0.002, 0.002, -0.003, 0.003]


def test_chapter29_pressure_hotspot_metrics_project_onto_beam_axis():
    import ch29_pressure_localization as localization

    target = np.zeros((5, 5), dtype=bool)
    target[2, 2] = True
    pressure = np.zeros((5, 5), dtype=float)
    pressure[3, 2] = 2.0
    geometry = {"planned_beam_axis_unit": [1.0, 0.0]}

    metrics = localization.pressure_hotspot_physical_metrics(pressure, target, [-2.0, 2.0, -2.0, 2.0], geometry)

    assert metrics["ct_frame_target_centroid_m"] == [0.0, 0.0]
    assert metrics["ct_frame_pressure_hotspot_m"] == [1.0, 0.0]
    assert np.isclose(metrics["ct_frame_pressure_hotspot_distance_m"], 1.0)
    assert np.isclose(metrics["ct_frame_pressure_hotspot_axis_offset_m"], 1.0)
    assert np.isclose(metrics["ct_frame_pressure_hotspot_cross_axis_offset_m"], 0.0)
    assert metrics["pressure_hotspot_is_postfocal"]
    assert not metrics["pressure_hotspot_is_prefocal"]


def test_chapter29_pressure_diagnostics_accept_projected_2d_fields():
    import ch29_pressure_diagnostics as diagnostics

    pressure = np.zeros((3, 3), dtype=float)
    pressure[1, 2] = 2.0
    target = np.zeros((3, 3), dtype=bool)
    target[1, 1] = True

    metrics = diagnostics.pressure_field_diagnostics(
        pressure,
        target,
        body_mask=np.ones_like(target, dtype=bool),
        frequency_hz=1.0e6,
        source_pressure_pa=1.0,
        source_scale=1.0,
        inertial_mi_threshold=1.9,
    )

    assert metrics["raw_peak_pressure_pa"] == 2.0
    assert metrics["raw_hotspot_x_index"] == 1.0
    assert metrics["raw_hotspot_y_index"] == 2.0
    assert metrics["raw_hotspot_z_index"] == 0.0
    assert metrics["target_centroid_z_index"] == 0.0


def test_chapter29_nonlinear_projection_expands_to_full_ct_frame():
    import ch29_theranostic_fwi_platforms as ch29

    crop = np.zeros((2, 2), dtype=float)
    crop[0, 0] = 1.0

    projected = ch29.project_to_ct_frame(
        crop,
        [-0.5, 0.5, -0.5, 0.5],
        (4, 4),
        [-1.5, 1.5, -1.5, 1.5],
    )
    projected_mask = ch29.project_to_ct_frame(
        crop,
        [-0.5, 0.5, -0.5, 0.5],
        (4, 4),
        [-1.5, 1.5, -1.5, 1.5],
        binary=True,
    )

    assert projected.shape == (4, 4)
    assert projected[1, 1] == 1.0
    assert np.count_nonzero(projected) == 1
    assert projected_mask.dtype == np.bool_
    assert projected_mask[1, 1]


def test_chapter29_fig05_westervelt_panel_masks_to_target_support():
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    pressure = np.zeros_like(target, dtype=float)
    pressure[1, 1, 1] = 2.0
    pressure[2, 2, 1] = 9.0

    display = ch29.nonlinear_target_pressure_volume({
        "target_mask": target,
        "body_mask": np.ones_like(target, dtype=bool),
        "westervelt_peak_pressure_pa": pressure,
    })

    assert display[1, 1, 1] == 2.0
    assert display[2, 2, 1] == 0.0
    assert float(np.max(display)) == 2.0


def test_chapter29_fig05_uses_nonlinear_beams_on_pressure_panels(monkeypatch):
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    nonlinear_points = np.asarray([[0.20, 0.30, 0.0], [0.25, 0.35, 0.0]], dtype=float)
    planned_points = np.asarray([[-0.20, -0.30, 0.0], [-0.25, -0.35, 0.0]], dtype=float)
    volume = target.astype(float)
    result = {
        "anatomy": "liver",
        "target_mask": target,
        "body_mask": np.ones_like(target, dtype=bool),
        "ct_hu": np.zeros_like(volume),
        "westervelt_peak_pressure_pa": volume,
        "multiparameter_fwi_score": volume,
        "reconstructed_delta_beta": volume,
        "cavitation_source_density": volume,
        "reconstructed_cavitation_density": volume,
        "nonlinear_fusion_score": volume,
        "therapy_points_m": nonlinear_points,
        "crop_bounds_index": [0, 2, 0, 2, 0, 2],
        "source_dimensions": [3, 3, 3],
        "source_spacing_m": [1.0, 1.0, 1.0],
        "wavelength_min_m": 1.0,
        "points_per_wavelength_min": 3.0,
        "min_points_per_wavelength": 2.0,
        "resolution_meets_min_ppw": True,
        "metrics": {
            "fwi": {"dice_equal_area": 1.0},
            "rayleigh_plesset_cavitation": {"dice_equal_area": 1.0},
            "fusion": {"dice_equal_area": 1.0},
        },
    }
    placement_target = np.zeros((5, 5), dtype=bool)
    placement_target[2, 2] = True
    placement = {
        "anatomy": "liver",
        "placement_ct_hu": np.zeros((5, 5), dtype=float),
        "placement_spacing_m": [1.0, 1.0],
        "placement_target_mask": placement_target,
        "placement_body_mask": np.ones((5, 5), dtype=bool),
        "placement_therapy_points_m": planned_points,
        "placement_imaging_points_m": np.asarray([[0.0, 0.0]], dtype=float),
        "placement_focus_m": [0.0, 0.0],
        "placement_skin_contact_m": [0.0, -1.0],
        "lesion_target": placement_target.astype(float),
        "exposure": placement_target.astype(float),
        "placement_metrics": {"min_body_clearance_m": 0.0},
    }
    calls = []

    def capture_beams(ax, therapy_points, focus):
        calls.append((np.asarray(therapy_points, dtype=float).copy(), np.asarray(focus, dtype=float).copy()))

    monkeypatch.setattr(ch29, "plot_beam_paths_2d", capture_beams)
    monkeypatch.setattr(ch29, "save_figure", lambda fig, path, dpi=None: None)

    ch29.render_nonlinear_3d([result], [placement])

    assert np.array_equal(calls[0][0], planned_points)
    assert np.array_equal(calls[1][0], planned_points[:, :2])
    assert np.array_equal(calls[2][0], nonlinear_points[:, :2])
