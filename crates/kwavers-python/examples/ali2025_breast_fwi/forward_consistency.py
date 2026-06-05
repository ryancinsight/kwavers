"""PyO3 wrappers for Rust-owned Ali 2025 observation consistency diagnostics."""

from __future__ import annotations

import numpy as np


def scaled_observation_residual_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    receiver_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    mask = _optional_bool_mask(receiver_mask)
    return dict(
        _kw().breast_fwi_scaled_observation_residual_metrics(
            _complex_cube(predicted),
            _complex_cube(observed),
            mask,
        )
    )


def source_channel_residual_diagnostics(
    predicted: np.ndarray,
    observed: np.ndarray,
    circumferential_elements: int,
    rows: int,
) -> dict[str, float | int]:
    return dict(
        _kw().breast_fwi_source_channel_residual_diagnostics(
            _complex_cube(predicted),
            _complex_cube(observed),
            int(circumferential_elements),
            int(rows),
        )
    )


def source_receiver_mask(
    observation_shape: tuple[int, int, int],
    circumferential_elements: int,
    rows: int,
) -> np.ndarray:
    return np.asarray(
        _kw().breast_fwi_source_receiver_mask(
            _shape3(observation_shape),
            int(circumferential_elements),
            int(rows),
        ),
        dtype=bool,
    )


def passive_receiver_mask(
    observation_shape: tuple[int, int, int],
    circumferential_elements: int,
    rows: int,
) -> np.ndarray:
    return np.asarray(
        _kw().breast_fwi_passive_receiver_mask(
            _shape3(observation_shape),
            int(circumferential_elements),
            int(rows),
        ),
        dtype=bool,
    )


def _kw():
    import pykwavers as kw

    return kw


def _complex_cube(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128)


def _optional_bool_mask(receiver_mask: np.ndarray | None) -> np.ndarray | None:
    if receiver_mask is None:
        return None
    mask = np.asarray(receiver_mask)
    if mask.dtype != np.bool_:
        raise ValueError("receiver_mask must be boolean")
    return mask


def _shape3(observation_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(observation_shape) != 3:
        raise ValueError(f"expected a 3-D observation shape, got {observation_shape}")
    return tuple(int(axis) for axis in observation_shape)
