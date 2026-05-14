from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))

from abdominal_fwi.model import (  # noqa: E402
    ApertureConfig,
    run_case,
)
from abdominal_fwi.operators import (  # noqa: E402
    build_fundamental_matrix,
    build_nonlinear_matrix,
    build_subharmonic_matrix,
)
from abdominal_fwi.preprocessing import prepare_ct_slice  # noqa: E402
from abdominal_fwi.regularization import laplacian_apply  # noqa: E402
from abdominal_fwi.simulation import simulate_westervelt_channels  # noqa: E402


def synthetic_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[:64, :64]
    organ = ((y - 32.0) / 21.0) ** 2 + ((x - 33.0) / 18.0) ** 2 <= 1.0
    target = ((y - 30.0) / 5.5) ** 2 + ((x - 36.0) / 4.5) ** 2 <= 1.0
    bone = ((y - 47.0) / 4.0) ** 2 + ((x - 23.0) / 8.0) ** 2 <= 1.0
    body = ((y - 32.0) / 27.0) ** 2 + ((x - 33.0) / 24.0) ** 2 <= 1.0

    label_slice = np.zeros((64, 64), dtype=np.int16)
    label_slice[organ] = 5
    label_slice[target] = 6
    label_slice[bone] = 4

    ct_slice = np.full((64, 64), -900.0, dtype=np.float32)
    ct_slice[body] = -80.0
    ct_slice[organ] = 60.0
    ct_slice[target] = 95.0
    ct_slice[bone] = 650.0

    speed_slice = np.full((64, 64), 343.0, dtype=np.float32)
    speed_slice[body] = 1480.0
    speed_slice[organ] = 1560.0
    speed_slice[target] = 1525.0
    speed_slice[bone] = 3000.0

    ct = np.stack([ct_slice - 5.0, ct_slice, ct_slice + 5.0])
    label = np.stack([label_slice, label_slice, label_slice])
    speed = np.stack([speed_slice, speed_slice, speed_slice])
    return ct, label, speed


def prepared_synthetic_slice(output_size: int = 36):
    ct, label, speed = synthetic_volume()
    return prepare_ct_slice(
        name="synthetic",
        title="synthetic abdominal target",
        ct_hu=ct,
        label=label,
        sound_speed_m_s=speed,
        focus_index=1,
        input_spacing_m=0.0015,
        organ_labels=(5,),
        target_labels=(6,),
        output_size=output_size,
        margin_voxels=10,
    )


def test_prepare_ct_slice_preserves_masks_and_physical_spacing():
    prepared = prepared_synthetic_slice(output_size=40)

    assert prepared.ct_hu.shape == (40, 40)
    assert prepared.sound_speed_m_s.shape == (40, 40)
    assert np.isclose(prepared.spacing_m, 0.0015 * 31.0 / 40.0)
    assert int(np.count_nonzero(prepared.target_mask)) == 140
    assert int(np.count_nonzero(prepared.organ_mask)) == 1453
    assert int(np.count_nonzero(prepared.imaging_mask)) > 1500
    assert float(np.std(prepared.sound_speed_m_s[prepared.imaging_mask])) > 30.0


def test_born_matrix_rows_are_normalized_and_geometry_sensitive():
    prepared = prepared_synthetic_slice(output_size=32)
    config = ApertureConfig(
        element_count=24,
        frequencies_hz=(220_000.0, 360_000.0),
        receiver_offsets=(6, 9),
        radius_margin_m=0.015,
        iterations=6,
    )

    active = prepared.imaging_mask
    matrix = build_fundamental_matrix(prepared, active, config)

    assert matrix.shape == (384, int(np.count_nonzero(active)))
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2.0e-7)
    assert float(np.std(matrix[:, 0])) > 0.01
    assert float(np.std(matrix[0, :])) > 0.001


def test_subharmonic_and_nonlinear_rows_are_distinct_channels():
    prepared = prepared_synthetic_slice(output_size=30)
    config = ApertureConfig(
        element_count=20,
        imaging_receiver_count=16,
        imaging_receiver_samples=4,
        frequencies_hz=(240_000.0, 420_000.0),
        receiver_offsets=(5, 8),
        iterations=8,
    )

    active = prepared.imaging_mask
    fundamental = build_fundamental_matrix(prepared, active, config)
    subharmonic = build_subharmonic_matrix(prepared, active, config)
    nonlinear = build_nonlinear_matrix(prepared, active, config)

    expected_rows = 20 * (2 + 4) * 2
    assert fundamental.shape == (expected_rows, int(np.count_nonzero(active)))
    assert subharmonic.shape == fundamental.shape
    assert nonlinear.shape == fundamental.shape
    assert np.allclose(np.linalg.norm(subharmonic, axis=1), 1.0, atol=2.0e-7)
    assert np.allclose(np.linalg.norm(nonlinear, axis=1), 1.0, atol=2.0e-7)
    assert not np.allclose(fundamental[0], subharmonic[0])
    assert not np.allclose(fundamental[0], nonlinear[0])


def test_laplacian_regularization_matches_four_neighbor_graph():
    mask = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)

    applied = laplacian_apply(mask, values)

    assert np.allclose(applied, [-3.0, -2.0, -11.0, 4.0, 12.0])


def test_westervelt_simulation_generates_harmonic_and_bubble_sources():
    prepared = prepared_synthetic_slice(output_size=28)
    lesion = prepared.target_mask
    config = ApertureConfig(
        element_count=16,
        imaging_receiver_count=16,
        imaging_receiver_samples=3,
        frequencies_hz=(240_000.0,),
        receiver_offsets=(4,),
        westervelt_frequency_hz=420_000.0,
        westervelt_source_pressure_pa=1.0e6,
        westervelt_cycles=6.0,
        westervelt_pml_cells=4,
    )

    channels = simulate_westervelt_channels(prepared, lesion, config)

    assert channels.metrics["nonlinear_forward_model"] == "2d_westervelt_fdtd"
    assert channels.fundamental_pressure_pa.shape == prepared.ct_hu.shape
    assert float(np.max(channels.fundamental_pressure_pa)) > 0.0
    assert float(np.max(channels.second_harmonic_pressure_pa)) > 0.0
    assert float(np.max(channels.subharmonic_source)) > 0.0
    assert float(np.mean(channels.subharmonic_source[lesion])) > 0.0
    assert float(np.max(channels.nonlinear_source[lesion])) == 1.0


def test_abdominal_fwi_recovers_targeting_and_lesion_state_contrast():
    prepared = prepared_synthetic_slice(output_size=34)
    config = ApertureConfig(
        element_count=40,
        frequencies_hz=(220_000.0, 360_000.0, 520_000.0),
        receiver_offsets=(10, 14, 18),
        radius_margin_m=0.018,
        iterations=18,
        regularization=8.0e-4,
        lesion_delta_c_m_s=-35.0,
    )

    result = run_case(prepared, config)

    assert result.targeting.metrics["measurements"] == 1080
    assert result.targeting.metrics["objective_reduction"] > 0.90
    assert result.targeting.metrics["pearson_correlation"] > 0.55
    assert result.targeting.metrics["nrmse"] < 0.85
    assert result.lesioning.metrics["objective_reduction"] > 0.90
    assert result.lesioning.metrics["equal_volume_dice"] > 0.35
    assert result.lesioning.metrics["lesion_cnr"] > 1.0
    assert result.subharmonic.metrics["objective_reduction"] > 0.90
    assert result.subharmonic.metrics["equal_volume_dice"] > 0.35
    assert result.subharmonic.metrics["lesion_cnr"] > 1.0
    assert result.nonlinear.metrics["objective_reduction"] > 0.90
    assert result.nonlinear.metrics["equal_volume_dice"] > 0.35
    assert result.nonlinear.metrics["lesion_cnr"] > 1.0
