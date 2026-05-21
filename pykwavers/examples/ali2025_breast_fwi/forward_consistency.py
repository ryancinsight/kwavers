"""PSTD observation versus frequency-domain forward consistency metrics."""

from __future__ import annotations

import numpy as np


def scaled_observation_residual_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    receiver_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    predicted, observed = paired_observation_cubes(predicted, observed)
    mask = observation_receiver_mask(predicted.shape, receiver_mask)
    residual_norm_sq = 0.0
    observed_norm_sq = 0.0
    max_abs_residual = 0.0
    row_residuals = []
    scales = []

    for frequency_index in range(predicted.shape[0]):
        for transmit_index in range(predicted.shape[1]):
            row_mask = mask[frequency_index, transmit_index, :]
            pred = predicted[frequency_index, transmit_index, row_mask]
            obs = observed[frequency_index, transmit_index, row_mask]
            scale = complex_row_scale(pred, obs)
            residual = scale * pred - obs
            residual_sq = float(np.vdot(residual, residual).real)
            observed_sq = float(np.vdot(obs, obs).real)
            residual_norm_sq += residual_sq
            observed_norm_sq += observed_sq
            max_abs_residual = max(max_abs_residual, float(np.max(np.abs(residual))))
            row_residuals.append((residual_sq / max(observed_sq, np.finfo(np.float64).eps)) ** 0.5)
            scales.append(scale)

    observed_norm = observed_norm_sq**0.5
    residual_norm = residual_norm_sq**0.5
    scale_magnitudes = np.abs(np.asarray(scales, dtype=np.complex128))
    row_residual_array = np.asarray(row_residuals, dtype=np.float64)
    return {
        "frequency_count": int(predicted.shape[0]),
        "transmission_count": int(predicted.shape[1]),
        "receiver_count": int(predicted.shape[2]),
        "row_count": int(predicted.shape[0] * predicted.shape[1]),
        "selected_receiver_count": int(np.count_nonzero(mask)),
        "observed_l2_norm": observed_norm,
        "scaled_residual_l2_norm": residual_norm,
        "normalized_l2_residual": residual_norm / max(observed_norm, np.finfo(np.float64).eps),
        "max_abs_scaled_residual": max_abs_residual,
        "row_normalized_l2_residual_mean": float(np.mean(row_residual_array)),
        "row_normalized_l2_residual_max": float(np.max(row_residual_array)),
        "source_scale_magnitude_min": float(np.min(scale_magnitudes)),
        "source_scale_magnitude_max": float(np.max(scale_magnitudes)),
    }


def source_channel_residual_diagnostics(
    predicted: np.ndarray,
    observed: np.ndarray,
    circumferential_elements: int,
    rows: int,
) -> dict[str, float | int]:
    predicted, observed = paired_observation_cubes(predicted, observed)
    active_mask = source_receiver_mask(predicted.shape, circumferential_elements, rows)
    passive_mask = np.logical_not(active_mask)

    full_metrics = scaled_observation_residual_metrics(predicted, observed)
    passive_metrics = scaled_observation_residual_metrics(predicted, observed, passive_mask)
    residual = scaled_residual_cube(predicted, observed)

    active_residual_sq = masked_l2_norm_sq(residual, active_mask)
    passive_residual_sq = masked_l2_norm_sq(residual, passive_mask)
    active_observed_sq = masked_l2_norm_sq(observed, active_mask)
    passive_observed_sq = masked_l2_norm_sq(observed, passive_mask)
    total_residual_sq = active_residual_sq + passive_residual_sq

    return {
        "active_receiver_count_per_row": int(rows),
        "passive_receiver_count_per_row": int(predicted.shape[2] - rows),
        "all_channel_normalized_l2_residual": float(full_metrics["normalized_l2_residual"]),
        "passive_only_normalized_l2_residual": float(
            passive_metrics["normalized_l2_residual"]
        ),
        "passive_only_scaled_residual_l2_norm": float(
            passive_metrics["scaled_residual_l2_norm"]
        ),
        "active_full_scale_residual_l2_norm": active_residual_sq**0.5,
        "passive_full_scale_residual_l2_norm": passive_residual_sq**0.5,
        "active_full_scale_observed_l2_norm": active_observed_sq**0.5,
        "passive_full_scale_observed_l2_norm": passive_observed_sq**0.5,
        "active_full_scale_normalized_l2_residual": active_residual_sq**0.5
        / max(active_observed_sq**0.5, np.finfo(np.float64).eps),
        "passive_full_scale_normalized_l2_residual": passive_residual_sq**0.5
        / max(passive_observed_sq**0.5, np.finfo(np.float64).eps),
        "active_full_scale_residual_energy_fraction": active_residual_sq
        / max(total_residual_sq, np.finfo(np.float64).eps),
        "passive_full_scale_residual_energy_fraction": passive_residual_sq
        / max(total_residual_sq, np.finfo(np.float64).eps),
    }


def scaled_residual_cube(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    predicted, observed = paired_observation_cubes(predicted, observed)
    residual = np.empty_like(predicted)
    for frequency_index in range(predicted.shape[0]):
        for transmit_index in range(predicted.shape[1]):
            pred = predicted[frequency_index, transmit_index, :]
            obs = observed[frequency_index, transmit_index, :]
            residual[frequency_index, transmit_index, :] = (
                complex_row_scale(pred, obs) * pred - obs
            )
    return residual


def source_receiver_mask(
    observation_shape: tuple[int, int, int],
    circumferential_elements: int,
    rows: int,
) -> np.ndarray:
    frequency_count, transmissions, receivers = validated_observation_shape(observation_shape)
    if circumferential_elements < 2:
        raise ValueError("circumferential_elements must be at least 2")
    if rows <= 0:
        raise ValueError("rows must be positive")
    if transmissions != circumferential_elements:
        raise ValueError(
            "transmission count must equal circumferential_elements for cylindrical firing"
        )
    expected_receivers = circumferential_elements * rows
    if receivers != expected_receivers:
        raise ValueError(
            f"receiver count mismatch: expected {expected_receivers}, got {receivers}"
        )

    mask = np.zeros((frequency_count, transmissions, receivers), dtype=bool)
    for transmit_index in range(transmissions):
        for row_index in range(rows):
            receiver_index = row_index * circumferential_elements + transmit_index
            mask[:, transmit_index, receiver_index] = True
    return mask


def passive_receiver_mask(
    observation_shape: tuple[int, int, int],
    circumferential_elements: int,
    rows: int,
) -> np.ndarray:
    active_mask = source_receiver_mask(observation_shape, circumferential_elements, rows)
    return np.logical_not(active_mask)


def complex_row_scale(predicted: np.ndarray, observed: np.ndarray) -> complex:
    pred = np.asarray(predicted, dtype=np.complex128).ravel()
    obs = np.asarray(observed, dtype=np.complex128).ravel()
    if pred.shape != obs.shape:
        raise ValueError(f"row shape mismatch: predicted {pred.shape}, observed {obs.shape}")
    if pred.size == 0:
        raise ValueError("observation row must not be empty")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(obs)):
        raise ValueError("observation row contains nonfinite complex values")
    denom = np.vdot(pred, pred)
    if float(denom.real) <= np.finfo(np.float64).eps:
        raise ValueError("predicted observation row has zero energy")
    return complex(np.vdot(pred, obs) / denom)


def masked_l2_norm_sq(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.vdot(selected, selected).real)


def paired_observation_cubes(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=np.complex128)
    obs = np.asarray(observed, dtype=np.complex128)
    if pred.ndim != 3 or obs.ndim != 3:
        raise ValueError(f"expected 3-D observation cubes, got {pred.shape} and {obs.shape}")
    if pred.shape != obs.shape:
        raise ValueError(f"observation shape mismatch: predicted {pred.shape}, observed {obs.shape}")
    if pred.size == 0:
        raise ValueError("observation cubes must not be empty")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(obs)):
        raise ValueError("observation cubes contain nonfinite complex values")
    return pred, obs


def observation_receiver_mask(
    observation_shape: tuple[int, int, int],
    receiver_mask: np.ndarray | None,
) -> np.ndarray:
    if receiver_mask is None:
        return np.ones(observation_shape, dtype=bool)
    raw = np.asarray(receiver_mask)
    if raw.dtype != np.bool_:
        raise ValueError("receiver_mask must be boolean")
    if raw.shape != observation_shape:
        raise ValueError(
            f"receiver_mask shape mismatch: expected {observation_shape}, got {raw.shape}"
        )
    if not np.all(np.any(raw, axis=2)):
        raise ValueError(
            "receiver_mask must select at least one receiver for every frequency/transmit row"
        )
    return raw.astype(bool, copy=False)


def validated_observation_shape(observation_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(observation_shape) != 3:
        raise ValueError(f"expected a 3-D observation shape, got {observation_shape}")
    frequency_count, transmissions, receivers = (int(axis) for axis in observation_shape)
    shape = (frequency_count, transmissions, receivers)
    if min(shape) <= 0:
        raise ValueError(f"observation dimensions must be positive, got {observation_shape}")
    return shape
