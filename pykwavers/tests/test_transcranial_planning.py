from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))

from transcranial_planning import data as planning_data  # noqa: E402
from transcranial_planning.data import Volume, load_default_brain_triplet, load_gbm_case, resample_volume  # noqa: E402
from transcranial_planning.data import brain_mask_from_ct  # noqa: E402
from transcranial_planning.metrics import normalized_mutual_information, registration_quality  # noqa: E402
from transcranial_planning.modality_bridge import (  # noqa: E402
    build_modality_bridge_plan,
    modality_bridge_manifest,
)
from transcranial_planning.registration import (  # noqa: E402
    _ritk_zyx_to_xyz,
    _xyz_to_ritk_zyx,
    affine_register_moving_to_fixed,
    register_triplet_with_ritk,
)
from transcranial_planning.simulation import (  # noqa: E402
    acoustic_observables,
    bbb_opening_from_subspots,
    gbm_subspot_plan,
    pennes_thermal_dose,
)
from transcranial_planning.transducer import (  # noqa: E402
    TransducerConfig,
    fibonacci_hemisphere,
    phase_correction_through_ct,
    phase_correction_through_skull,
)


def test_insightec_geometry_has_1024_elements_on_hemisphere_radius():
    config = TransducerConfig()
    positions = fibonacci_hemisphere(config)
    radii = np.linalg.norm(positions, axis=1)

    assert positions.shape == (1024, 3)
    assert np.allclose(radii, config.radius_m, rtol=0.0, atol=1.0e-12)
    assert np.all(positions[:, 2] < 0.0)


def test_phase_correction_depends_on_skull_path_length():
    skull = np.zeros((17, 17, 17), dtype=bool)
    skull[7:10, :, :] = True
    config = TransducerConfig(element_count=64)

    phase = phase_correction_through_skull(
        skull,
        spacing_m=(1.0e-3, 1.0e-3, 1.0e-3),
        target_index=(8, 8, 8),
        config=config,
        samples_per_ray=96,
    )

    assert phase.element_positions_m.shape == (64, 3)
    assert np.ptp(phase.skull_lengths_m) > 0.0
    assert np.ptp(phase.phases_rad) > 0.0
    assert np.all((phase.amplitude_weights > 0.0) & (phase.amplitude_weights <= 1.0))
    assert np.all(np.isfinite(phase.delays_s))


def test_ct_phase_correction_accounts_for_attenuation_and_reflection():
    ct = np.zeros((17, 17, 17), dtype=np.float32)
    ct[7:10, :, :] = 1600.0
    skull = ct > 300.0
    config = TransducerConfig(element_count=64)

    phase = phase_correction_through_ct(
        ct,
        spacing_m=(1.0e-3, 1.0e-3, 1.0e-3),
        target_index=(8, 8, 8),
        config=config,
        samples_per_ray=96,
        skull_mask=skull,
    )

    assert np.ptp(phase.skull_lengths_m) > 0.0
    assert np.ptp(phase.amplitude_weights) > 0.0
    assert float(phase.amplitude_weights.min()) < 1.0
    assert np.all(np.isfinite(phase.amplitude_weights))


def test_thermal_dose_inspects_values_and_forms_focal_lesion():
    shape = (13, 13, 13)
    pressure = np.zeros(shape, dtype=np.float32)
    pressure[6, 6, 6] = 2.0e6
    acoustic = acoustic_observables(pressure, frequency_hz=650.0e3)
    skull = np.zeros(shape, dtype=bool)
    brain = np.ones(shape, dtype=bool)

    thermal = pennes_thermal_dose(
        acoustic.intensity_w_m2,
        skull,
        brain,
        spacing_m=(1.0e-3, 1.0e-3, 1.0e-3),
        sonication_s=4.0,
        dt_s=0.1,
    )

    assert float(thermal.peak_temperature_c[6, 6, 6]) > 43.0
    assert float(thermal.cem43_min[6, 6, 6]) > float(thermal.cem43_min[0, 0, 0])
    assert thermal.lesion_mask[6, 6, 6]
    assert not thermal.lesion_mask[0, 0, 0]


def test_gbm_subspots_cover_nonempty_tumor_and_stay_inside_mask():
    tumor = np.zeros((15, 15, 15), dtype=bool)
    tumor[5:10, 5:10, 5:10] = True

    plan = gbm_subspot_plan(tumor, spacing_m=(1.0e-3, 1.0e-3, 1.0e-3), pitch_m=2.0e-3)

    assert plan.indices.shape[1] == 3
    assert plan.covered_fraction > 0.50
    assert np.all(tumor[plan.indices[:, 0], plan.indices[:, 1], plan.indices[:, 2]])


def test_bbb_opening_subspots_create_stable_permeability_without_ic_risk():
    tumor = np.zeros((17, 17, 17), dtype=bool)
    tumor[6:11, 6:11, 6:11] = True
    plan = gbm_subspot_plan(tumor, spacing_m=(1.0e-3, 1.0e-3, 1.0e-3), pitch_m=2.0e-3)

    result = bbb_opening_from_subspots(
        tumor,
        plan,
        spacing_m=(1.0e-3, 1.0e-3, 1.0e-3),
        mechanical_index=0.45,
        sonication_s=60.0,
        duty_cycle=0.02,
        focal_radius_m=2.0e-3,
    )

    opened_fraction = np.count_nonzero(result.opened_mask & tumor) / np.count_nonzero(tumor)
    assert opened_fraction > 0.50
    assert float(result.permeability[tumor].max()) > 0.50
    assert float(result.stable_cavitation_probability[tumor].max()) > 0.50
    assert float(result.inertial_cavitation_risk[tumor].max()) < 0.10


def test_local_upenn_gbm_sample_executes_segmentation_branch():
    case = load_gbm_case(shape=(16, 20, 16))

    assert case is not None
    ct, t1gd, flair, tumor, dataset, segmentation_space = case
    assert dataset in {"CFB-GBM", "UPenn-GBM", "RIRE-CT-segmentation"}
    assert segmentation_space in {"ct", "mri"}
    assert t1gd.data.shape == (16, 20, 16)
    assert flair.data.shape == (16, 20, 16)
    assert tumor.shape == (16, 20, 16)
    assert np.count_nonzero(tumor) > 0
    if segmentation_space == "mri":
        assert ct is None
    if segmentation_space == "ct":
        assert ct is not None


def test_ct_segmentation_path_is_first_class_for_bbb_planning():
    case = load_gbm_case(shape=(16, 20, 16))

    assert case is not None
    ct, _t1gd, _flair, tumor, _dataset, segmentation_space = case
    assert segmentation_space == "ct"
    assert ct is not None
    plan = gbm_subspot_plan(tumor, ct.spacing_m, pitch_m=3.0e-3)
    result = bbb_opening_from_subspots(tumor, plan, ct.spacing_m)
    assert plan.indices.shape[0] > 0
    assert float(result.permeability[tumor].max()) > 0.0


def test_modality_bridge_marks_ct_space_segmentation_as_same_subject_ready():
    paths = planning_data.discover_ct_segmentation_case()

    plan = build_modality_bridge_plan(paths)
    manifest = modality_bridge_manifest(paths)

    assert paths is not None
    assert plan.simulation_ready
    assert plan.planning_space == "ct"
    assert plan.skull_acoustics_same_subject
    assert "ct" in plan.available_modalities
    assert "segmentation" in plan.available_modalities
    assert any(action.action == "accept_ct_space_segmentation" and action.status == "ready" for action in plan.actions)
    ct_pair = next(requirement for requirement in plan.requirements if requirement.name == "ct_segmentation_pair")
    assert ct_pair.satisfied
    assert manifest["plan"]["simulation_scope"] == "ct_space_skull_acoustics_and_bbb_subspot_simulation"


def test_modality_bridge_keeps_upenn_mri_case_non_ct_backed():
    paths = planning_data.discover_upenn_gbm_case()

    plan = build_modality_bridge_plan(paths)

    assert paths is not None
    assert plan.simulation_ready
    assert plan.planning_space == "mri"
    assert not plan.skull_acoustics_same_subject
    assert "ct" in plan.missing_modalities
    assert any(action.action == "accept_mri_space_segmentation" for action in plan.actions)
    synthetic_ct = next(action for action in plan.actions if action.action == "external_synthetic_ct_candidate")
    assert synthetic_ct.reference == "cWDM"
    assert synthetic_ct.status == "external_required"
    assert "not synthesize CT" in synthetic_ct.boundary
    expected_suffixes = ("bridge\\synthetic_ct_cwdm.nii.gz", "bridge/synthetic_ct_cwdm.nii.gz")
    assert any(path.endswith(expected_suffixes) for path in synthetic_ct.artifact_paths)
    assert any(reference.name == "SLaM-DiMM" for reference in plan.references)


def test_ct_brain_mask_uses_skull_fill_not_background_hu():
    ct = np.zeros((15, 15, 3), dtype=np.float32)
    ct[3:12, 3, :] = 1000.0
    ct[3:12, 11, :] = 1000.0
    ct[3, 3:12, :] = 1000.0
    ct[11, 3:12, :] = 1000.0

    brain = brain_mask_from_ct(ct)

    assert brain[7, 7, 1]
    assert not brain[0, 0, 1]
    assert not brain[3, 7, 1]
    coords = np.argwhere(brain[:, :, 1])
    assert coords.min(axis=0).tolist() == [4, 4]
    assert coords.max(axis=0).tolist() == [10, 10]


def test_cfb_manifest_presence_does_not_use_gbm_fallback(monkeypatch, tmp_path):
    cfb_root = tmp_path / "cfb_gbm_sample"
    cfb_root.mkdir()
    monkeypatch.setattr(planning_data, "GBM_ROOT", cfb_root)

    sources = planning_data.dataset_sources()
    cfb_source = next(source for source in sources if source.name == "CFB-GBM")

    assert planning_data.discover_cfb_gbm_case(cfb_root) is None
    assert not cfb_source.present


def test_foreground_affine_refines_translation_with_mutual_information():
    fixed_data = np.zeros((21, 21, 21), dtype=np.float32)
    fixed_mask = np.zeros_like(fixed_data, dtype=bool)
    fixed_mask[3:18, 3:18, 3:18] = True
    fixed_data[fixed_mask] = 0.2
    fixed_data[7:12, 6:11, 8:13] = 1.0
    moving_data = np.zeros_like(fixed_data)
    moving_mask = fixed_mask.copy()
    moving_data[moving_mask] = 0.2
    moving_data[9:14, 5:10, 9:14] = 1.0
    fixed = Volume("fixed", fixed_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("fixed.nii.gz"))
    moving = Volume("moving", moving_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("moving.nii.gz"))

    result = affine_register_moving_to_fixed(
        moving,
        fixed,
        "moving in fixed space",
        strategy="foreground_extent",
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        axis_reflections=((1.0,), (1.0,), (1.0,)),
    )

    before = normalized_mutual_information(fixed_data, moving_data, mask=fixed_mask)
    after = normalized_mutual_information(fixed_data, result.moving_registered.data, mask=fixed_mask)
    moved_center = np.argwhere(result.moving_registered.data > 0.9).mean(axis=0)
    fixed_center = np.argwhere(fixed_data > 0.9).mean(axis=0)
    assert "nmi_translation_refinement" in result.method
    assert after > before
    assert np.allclose(moved_center, fixed_center)


def test_registration_quality_prefers_higher_nmi_and_lower_mse():
    baseline = registration_quality(ncc=0.1, nmi=1.1, mse=0.2)

    assert registration_quality(ncc=0.1, nmi=1.2, mse=0.2) > baseline
    assert registration_quality(ncc=0.1, nmi=1.1, mse=0.1) > baseline


def test_ritk_boundary_converts_xyz_to_zyx_and_back():
    xyz = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)

    zyx = _xyz_to_ritk_zyx(xyz)
    roundtrip = _ritk_zyx_to_xyz(zyx)

    assert zyx.shape == (5, 3, 2)
    assert float(zyx[4, 2, 1]) == float(xyz[1, 2, 4])
    assert np.array_equal(roundtrip, xyz)


def test_resample_volume_updates_affine_spacing_contract():
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    volume = Volume("source", data, (2.0e-3, 3.0e-3, 4.0e-3), affine, Path("source.nii.gz"))

    resampled = resample_volume(volume, (4, 4, 4), order=1)

    assert resampled.data.shape == (4, 4, 4)
    assert np.allclose(np.diag(resampled.affine)[:3], [1.0, 1.5, 2.0])
    assert resampled.spacing_m == (1.0e-3, 1.5e-3, 2.0e-3)


def test_affine_registration_resamples_moving_ct_to_fixed_mri_lattice():
    moving_data = np.zeros((7, 7, 7), dtype=np.float32)
    moving_data[2:5, 2:5, 2:5] = 1000.0
    fixed_data = np.zeros((7, 7, 7), dtype=np.float32)
    fixed_data[2:5, 2:5, 2:5] = 1.0
    fixed_affine = np.eye(4, dtype=np.float64)
    moving_affine = np.eye(4, dtype=np.float64)
    moving = Volume("ct", moving_data, (1.0e-3, 1.0e-3, 1.0e-3), moving_affine, Path("ct.nii.gz"))
    fixed = Volume("mri", fixed_data, (1.0e-3, 1.0e-3, 1.0e-3), fixed_affine, Path("mri.nii.gz"))

    result = affine_register_moving_to_fixed(moving, fixed, "registered CT", order=0)

    assert result.executed
    assert result.moving_registered.data.shape == fixed.data.shape
    assert result.moving_registered.spacing_m == fixed.spacing_m
    assert float(result.moving_registered.data[3, 3, 3]) == 1000.0
    assert result.nmi > 0.0
    assert result.edge_overlap >= 0.0


def test_foreground_extent_affine_selects_ap_reflection_when_masks_require_it():
    fixed_data = np.zeros((9, 9, 9), dtype=np.float32)
    fixed_data[2:5, 1:2, 3:6] = 1.0
    fixed_data[2:5, 2:3, 3:6] = 2.0
    fixed_data[2:5, 3:4, 3:6] = 3.0
    moving_data = np.zeros((9, 9, 9), dtype=np.float32)
    moving_data[2:5, 5:6, 3:6] = 3.0
    moving_data[2:5, 6:7, 3:6] = 2.0
    moving_data[2:5, 7:8, 3:6] = 1.0
    fixed = Volume("fixed", fixed_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("fixed.nii.gz"))
    moving = Volume("moving", moving_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("moving.nii.gz"))

    result = affine_register_moving_to_fixed(
        moving,
        fixed,
        "moving in fixed space",
        order=0,
        strategy="foreground_extent",
        fixed_mask=fixed_data > 0.5,
        moving_mask=moving_data > 0.5,
    )

    assert result.method == "foreground_extent_reflection_affine_resampling"
    assert np.array_equal(result.moving_registered.data > 0.5, fixed_data > 0.5)
    assert result.nmi > 1.0
    assert result.edge_overlap > 0.0


def test_foreground_extent_affine_can_forbid_ap_reflection_for_atlas():
    fixed_data = np.zeros((9, 9, 9), dtype=np.float32)
    fixed_data[2:5, 1:2, 3:6] = 1.0
    fixed_data[2:5, 2:3, 3:6] = 2.0
    fixed_data[2:5, 3:4, 3:6] = 3.0
    moving_data = np.zeros((9, 9, 9), dtype=np.float32)
    moving_data[2:5, 5:6, 3:6] = 3.0
    moving_data[2:5, 6:7, 3:6] = 2.0
    moving_data[2:5, 7:8, 3:6] = 1.0
    fixed = Volume("fixed", fixed_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("fixed.nii.gz"))
    moving = Volume("moving", moving_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("moving.nii.gz"))

    result = affine_register_moving_to_fixed(
        moving,
        fixed,
        "moving in fixed space",
        order=0,
        strategy="foreground_extent",
        fixed_mask=fixed_data > 0.5,
        moving_mask=moving_data > 0.5,
        axis_reflections=((-1.0, 1.0), (1.0,), (-1.0, 1.0)),
    )

    assert not np.array_equal(result.moving_registered.data, fixed_data)
    assert np.count_nonzero(result.moving_registered.data > 0.5) == np.count_nonzero(fixed_data > 0.5)


def test_atlas_like_affine_forbids_lr_and_si_reflections():
    fixed_data = np.zeros((9, 9, 9), dtype=np.float32)
    fixed_data[1:2, 2:5, 3:6] = 1.0
    fixed_data[2:3, 2:5, 3:6] = 2.0
    fixed_data[3:4, 2:5, 3:6] = 3.0
    moving_data = np.zeros((9, 9, 9), dtype=np.float32)
    moving_data[5:6, 2:5, 3:6] = 3.0
    moving_data[6:7, 2:5, 3:6] = 2.0
    moving_data[7:8, 2:5, 3:6] = 1.0
    fixed = Volume("fixed", fixed_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("fixed.nii.gz"))
    moving = Volume("moving", moving_data, (1.0e-3, 1.0e-3, 1.0e-3), np.eye(4), Path("moving.nii.gz"))

    result = affine_register_moving_to_fixed(
        moving,
        fixed,
        "moving in fixed space",
        order=0,
        strategy="foreground_extent",
        fixed_mask=fixed_data > 0.5,
        moving_mask=moving_data > 0.5,
        axis_reflections=((1.0,), (1.0,), (1.0,)),
    )

    assert not np.array_equal(result.moving_registered.data, fixed_data)
    assert np.count_nonzero(result.moving_registered.data > 0.5) == np.count_nonzero(fixed_data > 0.5)


def test_ritk_nifti_registration_uses_moving_output_not_fixed_return():
    pytest.importorskip("ritk")

    triplet = load_default_brain_triplet(shape=(16, 20, 16))
    result = register_triplet_with_ritk(triplet, max_iterations=2)

    assert result.executed
    assert result.message == "RITK NIfTI read and metric-guarded registration executed"
    assert result.t1_registered.data.shape == triplet.ct_hu.data.shape
    assert result.t1_registered.spacing_m == triplet.ct_hu.spacing_m
    assert np.allclose(result.t1_registered.affine, triplet.ct_hu.affine)
    assert np.allclose(result.atlas_registered.affine, triplet.ct_hu.affine)
    assert result.nmi_t1_after > 0.0
    assert result.nmi_atlas_after > 0.0
    assert result.mse_t1_after >= 0.0
    assert result.mse_atlas_after >= 0.0
    assert result.ncc_t1_after < 0.95
