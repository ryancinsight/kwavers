from __future__ import annotations

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

    assert chapters[27]["script"] == "ch27_seismic_fwi_brain_imaging.py"
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


def test_book_readme_links_bbb_and_hifu_markdown_chapters():
    readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")

    assert "(bbb_lifu_opening.md)" in readme
    assert "(hifu_transcranial_ablation.md)" in readme
    assert "(neuromodulation.md)" in readme
    assert "(seismic_fwi_brain_imaging.md)" in readme
    assert "(abdominal_histotripsy_fwi.md)" in readme
    assert "(theranostic_fwi_platforms.md)" in readme
    assert "(intravascular_ultrasound.md)" in readme
    assert (DOCS_DIR / "bbb_lifu_opening.md").is_file()
    assert (DOCS_DIR / "hifu_transcranial_ablation.md").is_file()
    assert (DOCS_DIR / "neuromodulation.md").is_file()
    assert (DOCS_DIR / "seismic_fwi_brain_imaging.md").is_file()
    assert (DOCS_DIR / "abdominal_histotripsy_fwi.md").is_file()
    assert (DOCS_DIR / "theranostic_fwi_platforms.md").is_file()
    assert (DOCS_DIR / "intravascular_ultrasound.md").is_file()


def test_chapter29_layout_helpers_report_skin_and_helmet_clearance():
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
    assert "helmet clearance 15.0 mm" == ch29.placement_label(brain)
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
    assert all("FWI" not in title for _, title in ch29.RECONSTRUCTION_CHANNELS)
    assert all("FWI" not in title for _, _, title in ch29.RECONSTRUCTION_FIGURE_COLUMNS)


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

    assert default_grids == {"brain": 48, "kidney": 52, "liver": 52}

    monkeypatch.setenv("KWAVERS_CH29_NONLINEAR_GRID", "40")
    assert {case["name"]: ch29.nonlinear_grid_size(case) for case in ch29.CASES} == {
        "brain": 40,
        "kidney": 40,
        "liver": 40,
    }

    monkeypatch.setenv("KWAVERS_CH29_KIDNEY_NONLINEAR_GRID", "56")
    kidney = next(case for case in ch29.CASES if case["name"] == "kidney")

    assert ch29.nonlinear_grid_size(kidney) == 56


def test_chapter29_ct_context_draws_transducer_locations():
    import matplotlib.pyplot as plt

    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((5, 5), dtype=bool)
    target[2, 2] = True
    result = {
        "anatomy": "kidney",
        "device_model": "histosonics_like_256_element_skin_coupled_arc",
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
    linear = {
        "anatomy": "brain",
        "fused_reconstruction": linear_target,
        "active_lesion_reconstruction": linear_target,
        "exposure": linear_target,
        "spacing_m": 1.0,
        "focus_m": (0.0, 0.0),
        "therapy_x_m": np.asarray([-1.0, 1.0]),
        "therapy_y_m": np.asarray([0.0, 0.0]),
    }
    nonlinear = {
        "anatomy": "brain",
        "target_mask": target,
        "crop_bounds_index": [0, 3, 0, 3, 0, 2],
        "source_dimensions": [4, 4, 3],
        "source_spacing_m": [1.0, 1.0, 1.0],
        "westervelt_peak_pressure_pa": target.astype(float),
        "multiparameter_fwi_score": nonlinear_fusion,
        "reconstructed_cavitation_density": nonlinear_fusion,
        "nonlinear_fusion_score": nonlinear_fusion,
        "therapy_points_m": np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "fwi_objective_history": np.asarray([2.0, 1.25]),
        "cavitation_objective_history": np.asarray([1.0, 0.8]),
    }

    comparison = ch29.build_controlled_comparison([linear], [nonlinear])[0]

    assert comparison["common_grid_shape"] == [4, 4]
    assert comparison["geometry"]["common_target_voxels"] == 2
    assert comparison["comparison_metrics"]["linear_fusion"]["dice_equal_area"] == 1.0
    assert comparison["objective_history"]["nonlinear_fwi"] == [2.0, 1.25]
    assert "linear fusion Dice" in comparison["technical_explanation"]
