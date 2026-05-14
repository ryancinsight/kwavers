from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))

import ch27_seismic_fwi_brain_imaging as ch27  # noqa: E402
from seismic_fwi.roi import centroid_bounds, stack_roi_metadata  # noqa: E402


def test_synthetic_data_tensor_preserves_source_offset_frequency_ordering():
    result = {
        "synthetic_data": np.arange(24.0),
        "element_count": 4,
        "receiver_offsets": [2, 3],
        "frequencies_hz": [250_000.0, 400_000.0, 650_000.0],
        "harmonic_count": 1,
    }

    tensor = ch27.synthetic_data_tensor(result)

    assert tensor.shape == (4, 2, 3, 1)
    assert float(tensor[0, 0, 0, 0]) == 0.0
    assert float(tensor[0, 1, 2, 0]) == 5.0
    assert float(tensor[3, 1, 2, 0]) == 23.0


def test_synthetic_data_tensor_rejects_dimension_mismatch():
    result = {
        "synthetic_data": np.arange(23.0),
        "element_count": 4,
        "receiver_offsets": [2, 3],
        "frequencies_hz": [250_000.0, 400_000.0, 650_000.0],
        "harmonic_count": 1,
    }

    with pytest.raises(ValueError, match="synthetic_data size"):
        ch27.synthetic_data_tensor(result)


def test_synthetic_data_tensor_preserves_harmonic_channel_ordering():
    result = {
        "synthetic_data": np.arange(48.0),
        "element_count": 4,
        "receiver_offsets": [2, 3],
        "frequencies_hz": [250_000.0, 400_000.0, 650_000.0],
        "harmonic_count": 2,
    }

    tensor = ch27.synthetic_data_tensor(result)

    assert tensor.shape == (4, 2, 3, 2)
    assert float(tensor[0, 0, 0, 0]) == 0.0
    assert float(tensor[0, 0, 0, 1]) == 1.0
    assert float(tensor[0, 1, 2, 1]) == 11.0
    assert float(tensor[3, 1, 2, 1]) == 47.0


def test_hemispherical_projection_stays_inside_aperture_and_varies_radially():
    x, y = ch27.hemispherical_projection_mm(32, 110.0)
    radius = np.sqrt(x * x + y * y)

    assert x.shape == (32,)
    assert y.shape == (32,)
    assert np.all(radius <= 110.0)
    assert float(radius[0]) > float(radius[-1])
    assert float(np.ptp(radius)) > 50.0


def test_visible_reconstruction_requires_objective_and_contrast():
    visible = ch27.visible_reconstruction(
        {
            "objective_reduction_fraction": 0.60,
            "target_dynamic_range_m_s": 100.0,
            "reconstruction_dynamic_range_m_s": 40.0,
        }
    )
    weak_objective = ch27.visible_reconstruction(
        {
            "objective_reduction_fraction": 0.20,
            "target_dynamic_range_m_s": 100.0,
            "reconstruction_dynamic_range_m_s": 40.0,
        }
    )
    weak_contrast = ch27.visible_reconstruction(
        {
            "objective_reduction_fraction": 0.60,
            "target_dynamic_range_m_s": 100.0,
            "reconstruction_dynamic_range_m_s": 10.0,
        }
    )

    assert visible is True
    assert weak_objective is False
    assert weak_contrast is False


def test_regularized_fwi_display_is_mask_local_and_reconstruction_derived():
    reconstruction = np.full((5, 5), 1540.0)
    reconstruction[2, 2] = 1580.0
    reconstruction[0, 0] = 1200.0
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    display = ch27.regularized_fwi_display(reconstruction, mask)

    assert display.shape == reconstruction.shape
    assert np.array_equal(display[~mask], reconstruction[~mask])
    assert 1540.0 < float(display[2, 2]) < 1580.0
    assert float(np.ptp(display[mask])) > 0.0


def test_centroid_bounds_are_centered_and_clamped():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:7] = True
    edge_mask = np.zeros((3, 3), dtype=bool)
    edge_mask[0, 0] = True

    bounds = centroid_bounds(mask, spacing_m=0.002, half_width_mm=3.0)
    edge_bounds = centroid_bounds(edge_mask, spacing_m=0.002, half_width_mm=6.0)

    assert bounds.as_list() == [2, 7, 3, 8]
    assert edge_bounds.as_list() == [0, 3, 0, 3]


def test_centroid_roi_metadata_records_slice_region():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:7] = True
    empty = np.zeros((8, 10), dtype=bool)
    stack = [
        (
            {
                "brain_mask": empty,
                "spacing_m": 0.002,
                "source_slice_index": 9,
            },
            {"pearson_correlation": 0.0},
        ),
        (
            {
                "brain_mask": mask,
                "spacing_m": 0.002,
                "source_slice_index": 11,
            },
            {"pearson_correlation": 0.9},
        )
    ]

    metadata = stack_roi_metadata(stack, half_width_mm=3.0)

    assert metadata["half_width_mm"] == 3.0
    assert metadata["slice_indices"] == [11]
    assert metadata["skipped_empty_slice_indices"] == [9]
    assert metadata["regions"][0]["bounds_voxels"] == [2, 7, 3, 8]


def test_volume_result_is_sliced_without_running_independent_2d_inversions():
    shape = (5, 6, 4)
    target = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1500.0
    recon = target + 1.0
    migration = target - 2.0
    mask = np.ones(shape, dtype=bool)
    volume_result = {
        "ct_hu": target - 1500.0,
        "target_sound_speed_m_s": target,
        "initial_sound_speed_m_s": np.full(shape, 1540.0),
        "migration_sound_speed_m_s": migration,
        "reconstruction_sound_speed_m_s": recon,
        "enhanced_reconstruction_sound_speed_m_s": recon + 0.5,
        "brain_mask": mask,
        "skull_mask": np.zeros(shape, dtype=bool),
        "spacing_m": 0.002,
        "source_volume_index": 2,
        "source_slice_index": 20,
        "synthetic_data": np.arange(8.0),
        "element_count": 2,
        "receiver_offsets": [1],
        "frequencies_hz": [250_000.0, 500_000.0],
        "harmonic_count": 2,
    }

    sliced = ch27.slice_volume_result(
        volume_result,
        1,
        {"objective_reduction_fraction": 0.75},
    )

    assert sliced["target_sound_speed_m_s"].shape == (5, 6)
    assert np.array_equal(sliced["target_sound_speed_m_s"], target[:, :, 1])
    assert sliced["source_volume_index"] == 1
    assert sliced["source_slice_index"] == 1
    assert sliced["slice_offset_m"] == pytest.approx(-0.001)
    assert sliced["metrics"]["active_voxels"] == 30
    assert sliced["metrics"]["pearson_correlation"] == pytest.approx(1.0)
