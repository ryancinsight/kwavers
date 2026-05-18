from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))

from transcranial_planning.scene import CANONICAL_BRAIN_SCENE  # noqa: E402
from transcranial_planning.tfuscapes import (  # noqa: E402
    DEFAULT_TFUSCAPES_CASE,
    TFUSCAPES_BODY_HU_THRESHOLD,
    compare_tfuscapes_case_to_scene,
    load_tfuscapes_case,
    map_case_to_scene,
    run_tfuscapes_skull_adaptive_benchmark,
    write_tfuscapes_ct_nifti,
)


def test_tfuscapes_case_loader_extracts_minimal_paper_fields(tmp_path):
    path = write_fixture_case(tmp_path)

    case = load_tfuscapes_case(path)
    mapping = map_case_to_scene(case)
    comparison = compare_tfuscapes_case_to_scene(case, mapping)

    assert case.ct_hu.shape == (9, 9, 9)
    assert case.pressure_pa.shape == case.ct_hu.shape
    assert case.transducer_indices.shape == (5, 3)
    assert mapping.target_index == (4, 4, 4)
    assert comparison["fields"]["ct"]["shape"] == [9, 9, 9]
    assert comparison["fields"]["pmap"]["max"] == 10.0
    assert comparison["fields"]["tr_coords"]["shape"] == [5, 3]
    assert comparison["acceptance"]["target_from_pressure_peak_inside_brain_support"]


def test_tfuscapes_mapping_fits_actual_transducer_indices_to_shared_scene_radius(tmp_path):
    path = write_fixture_case(tmp_path)
    case = load_tfuscapes_case(path)

    mapping = map_case_to_scene(case, CANONICAL_BRAIN_SCENE)

    radii = np.linalg.norm(mapping.transducer_points_m, axis=1)
    assert np.isclose(np.median(radii), CANONICAL_BRAIN_SCENE.transducer.radius_m)
    assert mapping.spacing_m == (0.0375, 0.0375, 0.0375)
    assert mapping.cap_angle_max_deg > mapping.cap_angle_min_deg


def test_tfuscapes_end_to_end_path_reuses_existing_benchmark_runner(tmp_path):
    path = write_fixture_case(tmp_path)
    calls = []

    def fake_runner(ct_path, **kwargs):
        calls.append((ct_path, kwargs))
        assert Path(ct_path).exists()
        assert kwargs["kwargs_overrides"]["body_hu_threshold"] == TFUSCAPES_BODY_HU_THRESHOLD
        assert kwargs["kwargs_overrides"]["target_fraction_xyz"] == (0.5, 0.5, 0.5)
        return {
            "benchmark_model": "ct_conditioned_skull_aware_aperture_vs_uncorrected_baseline",
            "frequency_hz": 650_000.0,
            "target_peak_pa": 1.0e6,
            "focus_index": (2, 2, 2),
            "active_elements": np.array([True, False, True]),
            "reference_pressure_pa": np.ones((5, 5, 5), dtype=np.float32),
            "baseline_pressure_pa": np.full((5, 5, 5), 0.5, dtype=np.float32),
            "placement": {
                "aperture_diameter_m": 0.120,
                "radius_of_curvature_m": 0.150,
                "mean_skull_length_m": 0.006,
                "mean_amplitude_weight": 0.7,
            },
            "metrics": {
                "relative_l2": 0.2,
                "focal_position_error_m": 0.001,
                "max_pressure_error_percent": 9.0,
            },
            "paper_structural_comparison": {"reference_setup": "TFUScapes"},
        }

    result = run_tfuscapes_skull_adaptive_benchmark(
        path,
        work_dir=tmp_path,
        runner=fake_runner,
    )

    assert len(calls) == 1
    assert result["case"]["dataset"]["manifest_text"] == DEFAULT_TFUSCAPES_CASE.manifest_text
    assert result["benchmark"]["active_element_count"] == 2
    assert result["output_comparison"]["all_fields_are_rank_3"]
    assert result["output_comparison"]["paper_peak_index"] == [4, 4, 4]


def test_tfuscapes_ct_nifti_preserves_derived_spacing_contract(tmp_path):
    path = write_fixture_case(tmp_path)
    case = load_tfuscapes_case(path)
    mapping = map_case_to_scene(case)
    nifti_path = write_tfuscapes_ct_nifti(case, mapping, tmp_path / "case_ct.nii.gz")

    import nibabel as nib

    image = nib.load(str(nifti_path))
    assert image.shape == (9, 9, 9)
    assert np.allclose(image.header.get_zooms()[:3], (37.5, 37.5, 37.5))


def write_fixture_case(tmp_path: Path) -> Path:
    ct = np.zeros((9, 9, 9), dtype=np.float32)
    ct[2:7, 2:7, 2:7] = 40.0
    ct[1:8, 1, 1:8] = 900.0
    ct[1:8, 7, 1:8] = 900.0
    pmap = np.zeros_like(ct)
    pmap[4, 4, 4] = 10.0
    tr_coords = np.asarray(
        [
            [4, 8, 4],
            [8, 4, 4],
            [0, 4, 4],
            [4, 4, 8],
            [4, 4, 0],
        ],
        dtype=np.int64,
    )
    path = tmp_path / "case.npz"
    np.savez(path, ct=ct, pmap=pmap, tr_coords=tr_coords)
    return path
