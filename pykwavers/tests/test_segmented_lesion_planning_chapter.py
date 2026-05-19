from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
DOCS_DIR = BOOK_DIR.parents[2] / "docs" / "book"
sys.path.insert(0, str(BOOK_DIR))

from segmented_lesion_planning.figures import write_metrics  # noqa: E402
from segmented_lesion_planning.dense_field import dense_acceptance_key  # noqa: E402
from segmented_lesion_planning.liver_dataset import load_lits_liver_planning_grid  # noqa: E402
from segmented_lesion_planning.phantom import build_segmented_therapy_phantom  # noqa: E402
from segmented_lesion_planning.solver import desired_spot_profile, optimize_transducer_layout  # noqa: E402
from segmented_lesion_planning.types import HybridPlanConfig, Tissue  # noqa: E402


def compact_config() -> HybridPlanConfig:
    return HybridPlanConfig(
        element_count=10,
        candidate_angles_deg=(-170.0, -130.0, -90.0, -50.0, -10.0, 30.0, 70.0, 110.0, 150.0),
        max_target_points=36,
        max_avoid_points=24,
        max_normal_points=48,
        max_sidelobe_points=0,
        ray_samples=48,
        ridge=2.5e-3,
        crossfire_aperture_count=1,
        hotspot_refinement_rounds=0,
    )


@pytest.fixture(scope="module")
def compact_plan() -> tuple[object, dict[str, object]]:
    grid = build_segmented_therapy_phantom(n=80, spacing_m=1.4e-3)
    result = optimize_transducer_layout(grid, compact_config())
    return grid, result


def test_segmented_phantom_contains_required_planning_labels():
    grid = build_segmented_therapy_phantom(n=80, spacing_m=1.4e-3)

    assert grid.labels.shape == (80, 80)
    assert np.count_nonzero(grid.mask(Tissue.TUMOR)) > 100
    assert np.count_nonzero(grid.mask(Tissue.AVOID)) > 100
    assert np.count_nonzero(grid.mask(Tissue.BONE)) > 100
    assert np.count_nonzero(grid.mask(Tissue.FAT)) > 100
    assert np.count_nonzero(grid.mask(Tissue.NORMAL)) > 1000
    assert np.count_nonzero(grid.mask(Tissue.AIR) & grid.body_mask) == 0


def test_lits_liver_dataset_maps_native_labels_and_ct_hazards():
    repo_root = BOOK_DIR.parents[2]
    ct_path = repo_root / "data" / "lits17_sample" / "volume-0.nii"
    seg_path = repo_root / "data" / "lits17_sample" / "segmentation-0.nii"

    grid, metadata = load_lits_liver_planning_grid(ct_path, seg_path, output_size=96)

    assert metadata["source"] == "LiTS17 sample liver CT"
    assert metadata["segmentation_labels"] == {"normal": 1, "tumor": 2}
    assert metadata["target_rule"] == "largest connected label-2 component on selected slice"
    assert metadata["slice_index"] == 58
    assert grid.labels.shape == (96, 96)
    assert grid.mask(Tissue.TUMOR)[grid.index_from_point(grid.centroid(Tissue.TUMOR))]
    assert np.count_nonzero(grid.mask(Tissue.TUMOR)) > 0
    assert np.count_nonzero(grid.mask(Tissue.NORMAL)) > 1000
    assert np.count_nonzero(grid.mask(Tissue.AVOID)) > 0
    assert np.count_nonzero(grid.mask(Tissue.BONE)) > 0
    assert np.count_nonzero(grid.mask(Tissue.FAT)) > 0
    assert np.count_nonzero(grid.mask(Tissue.AIR)) > 0
    assert np.count_nonzero(grid.mask(Tissue.AIR) & grid.body_mask) == 0


def test_desired_spot_profile_is_shape_controlled_by_config():
    target = np.asarray([0.012, -0.007], dtype=float)
    config = HybridPlanConfig(spot_angle_deg=0.0, spot_major_axis_m=0.012, spot_minor_axis_m=0.004)
    points = np.asarray(
        [
            target,
            target + np.asarray([0.006, 0.0]),
            target + np.asarray([0.0, 0.006]),
        ],
        dtype=float,
    )

    profile = desired_spot_profile(points, target, config)

    assert np.isclose(profile[0], 1.0)
    assert profile[1] > profile[2]


def test_dense_acceptance_prefers_target_dominant_plan():
    rejected = {
        "target_dominant": False,
        "body_sidelobe_peak_ratio": 1.05,
        "body_sidelobe_p99_ratio": 0.20,
        "tumor_coverage_fraction": 1.0,
        "tumor_mean_intensity": 0.9,
        "protected_peak_ratio": 0.05,
        "normal_mean_ratio": 0.04,
    }
    accepted = {
        "target_dominant": True,
        "body_sidelobe_peak_ratio": 0.95,
        "body_sidelobe_p99_ratio": 0.40,
        "tumor_coverage_fraction": 0.7,
        "tumor_mean_intensity": 0.5,
        "protected_peak_ratio": 0.20,
        "normal_mean_ratio": 0.10,
    }

    assert dense_acceptance_key(accepted) > dense_acceptance_key(rejected)


def test_hybrid_optimizer_selects_access_path_and_suppresses_protected_peak(compact_plan):
    _, result = compact_plan
    summary = result["summary"]
    candidate_metrics = [candidate["metrics"] for candidate in result["candidates"]]

    assert summary["candidate_count"] == 9
    assert summary["selected_score"] == max(metric["score"] for metric in candidate_metrics)
    assert summary["protected_peak_ratio"] < 0.15
    assert summary["air_path_fraction"] < 0.01
    assert summary["bone_path_fraction"] == 0.0
    assert any(metric["bone_path_fraction"] > 0.10 for metric in candidate_metrics)
    assert any(metric["air_path_fraction"] > 0.10 for metric in candidate_metrics)


def test_metrics_writer_exports_selected_plan_contract(tmp_path: Path, compact_plan):
    grid, result = compact_plan
    metrics_path = write_metrics(grid, result, [], tmp_path / "metrics.json")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert payload["chapter"] == 32
    assert payload["dataset"]["source"] == "analytic segmented phantom"
    assert payload["selected_aperture"]["element_count"] == 10
    assert payload["segmentation_voxels"]["tumor"] > 100
    assert payload["summary"]["protected_peak_ratio"] < 0.15
    assert len(payload["candidate_metrics"]) == 9


def test_generated_liver_metrics_record_target_dominant_focus():
    payload = json.loads((DOCS_DIR / "figures" / "ch32" / "metrics.json").read_text(encoding="utf-8"))
    summary = payload["summary"]

    assert summary["target_dominant"] is True
    assert summary["body_sidelobe_peak_ratio"] == 0.7395404024847666
    assert summary["body_sidelobe_p99_ratio"] == 0.3297347520675772
    assert summary["tumor_coverage_fraction"] == 0.7837837837837838
    assert summary["protected_peak_ratio"] == 0.2958651403757349
