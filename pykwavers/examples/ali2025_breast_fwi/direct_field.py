"""Homogeneous direct-field diagnostics for Ali 2025 PSTD parity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .discrete_green import (
    pstd_periodic_observation_cube,
    source_kappa_filtered_observation_cube,
)
from .forward_consistency import (
    complex_row_scale,
    scaled_observation_residual_metrics,
    source_channel_residual_diagnostics,
)
from .source_excitation import source_excitation_diagnostics


def homogeneous_direct_field_probe(
    kw: Any,
    homogeneous_sound_speed_m_s: np.ndarray,
    array: Any,
    frequencies_hz: Sequence[float],
    dataset_config: Any,
) -> dict[str, Any]:
    observed_dataset = kw.generate_breast_fwi_pstd_dataset(
        homogeneous_sound_speed_m_s,
        array,
        list(frequencies_hz),
        dataset_config,
    )
    observed = np.asarray(observed_dataset["observed_pressure"], dtype=np.complex128)
    reference_speed = float(np.median(homogeneous_sound_speed_m_s))
    predicted = direct_green_observation_cube(
        np.asarray(array.elements(), dtype=np.float64),
        int(array.circumferential_elements),
        int(array.rows),
        frequencies_hz,
        reference_speed,
        float(dataset_config.spacing_m),
    )
    filtered = source_kappa_filtered_observation_cube(
        np.asarray(array.elements(), dtype=np.float64),
        int(array.circumferential_elements),
        int(array.rows),
        frequencies_hz,
        reference_speed,
        float(dataset_config.spacing_m),
        homogeneous_sound_speed_m_s.shape,
        float(dataset_config.time_step_s),
    )
    point_diagnostics = direct_field_diagnostics(
        predicted,
        observed,
        frequencies_hz,
        float(dataset_config.source_amplitude_pa),
        float(dataset_config.time_step_s),
        observed_dataset["time_steps_per_frequency"],
        observed_dataset["frequency_bin_start_steps_per_frequency"],
        int(array.circumferential_elements),
        int(array.rows),
        np.asarray(array.elements(), dtype=np.float64),
    )
    filtered_diagnostics = direct_field_diagnostics(
        filtered,
        observed,
        frequencies_hz,
        float(dataset_config.source_amplitude_pa),
        float(dataset_config.time_step_s),
        observed_dataset["time_steps_per_frequency"],
        observed_dataset["frequency_bin_start_steps_per_frequency"],
        int(array.circumferential_elements),
        int(array.rows),
        np.asarray(array.elements(), dtype=np.float64),
    )
    periodic = pstd_periodic_observation_cube(
        np.asarray(array.elements(), dtype=np.float64),
        int(array.circumferential_elements),
        int(array.rows),
        frequencies_hz,
        reference_speed,
        float(dataset_config.spacing_m),
        homogeneous_sound_speed_m_s.shape,
        float(dataset_config.time_step_s),
        observed_dataset["time_steps_per_frequency"],
        observed_dataset["frequency_bin_start_steps_per_frequency"],
        float(dataset_config.source_amplitude_pa),
    )
    periodic_diagnostics = direct_field_diagnostics(
        periodic,
        observed,
        frequencies_hz,
        float(dataset_config.source_amplitude_pa),
        float(dataset_config.time_step_s),
        observed_dataset["time_steps_per_frequency"],
        observed_dataset["frequency_bin_start_steps_per_frequency"],
        int(array.circumferential_elements),
        int(array.rows),
        np.asarray(array.elements(), dtype=np.float64),
    )
    return {
        **point_diagnostics,
        "source_kappa_filtered": filtered_diagnostics,
        "source_kappa_filtered_residual_delta": float(
            filtered_diagnostics["normalized_l2_residual"]
            - point_diagnostics["normalized_l2_residual"]
        ),
        "pstd_periodic": periodic_diagnostics,
        "pstd_periodic_residual_delta": float(
            periodic_diagnostics["normalized_l2_residual"]
            - point_diagnostics["normalized_l2_residual"]
        ),
    }


def direct_green_observation_cube(
    elements_xyz_m: np.ndarray,
    circumferential_elements: int,
    rows: int,
    frequencies_hz: Sequence[float],
    sound_speed_m_s: float,
    spacing_m: float,
) -> np.ndarray:
    elements = validated_elements(elements_xyz_m, circumferential_elements, rows)
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be positive and finite")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    min_distance = 0.5 * spacing_m
    output = np.empty(
        (len(frequencies_hz), circumferential_elements, elements.shape[0]),
        dtype=np.complex128,
    )
    for frequency_index, frequency_hz in enumerate(frequencies_hz):
        if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("frequencies_hz entries must be positive and finite")
        wavenumber = 2.0 * np.pi * float(frequency_hz) / sound_speed_m_s
        for transmit_index in range(circumferential_elements):
            source_indices = [
                row * circumferential_elements + transmit_index for row in range(rows)
            ]
            output[frequency_index, transmit_index, :] = np.sum(
                [
                    outgoing_green(elements[source_index], elements, wavenumber, min_distance)
                    for source_index in source_indices
                ],
                axis=0,
            )
    return output


def direct_field_diagnostics(
    predicted: np.ndarray,
    observed: np.ndarray,
    frequencies_hz: Sequence[float],
    source_amplitude_pa: float,
    time_step_s: float,
    time_steps_per_frequency: Sequence[int],
    frequency_bin_start_steps_per_frequency: Sequence[int],
    circumferential_elements: int,
    rows: int,
    elements_xyz_m: np.ndarray,
) -> dict[str, Any]:
    residual = scaled_observation_residual_metrics(predicted, observed)
    channels = source_channel_residual_diagnostics(
        predicted,
        observed,
        circumferential_elements,
        rows,
    )
    excitation = source_excitation_diagnostics(
        predicted,
        observed,
        frequencies_hz,
        source_amplitude_pa,
        time_step_s,
        time_steps_per_frequency,
        frequency_bin_start_steps_per_frequency,
    )
    passive = passive_pair_errors(
        predicted,
        observed,
        elements_xyz_m,
        circumferential_elements,
        rows,
    )
    return {
        "normalized_l2_residual": float(residual["normalized_l2_residual"]),
        "row_normalized_l2_residual_mean": float(residual["row_normalized_l2_residual_mean"]),
        "passive_only_normalized_l2_residual": float(
            channels["passive_only_normalized_l2_residual"]
        ),
        "source_scale_magnitude_coefficient_of_variation": float(
            excitation["max_source_scale_magnitude_coefficient_of_variation"]
        ),
        "source_scale_phase_span_rad": float(excitation["max_source_scale_phase_span_rad"]),
        **passive,
    }


def passive_pair_errors(
    predicted: np.ndarray,
    observed: np.ndarray,
    elements_xyz_m: np.ndarray,
    circumferential_elements: int,
    rows: int,
) -> dict[str, float | int]:
    elements = validated_elements(elements_xyz_m, circumferential_elements, rows)
    phase_errors = []
    log_amplitude_errors = []
    ranges = []
    for frequency_index in range(predicted.shape[0]):
        for transmit_index in range(predicted.shape[1]):
            scale = complex_row_scale(predicted[frequency_index, transmit_index], observed[
                frequency_index,
                transmit_index,
            ])
            for receiver_index in range(predicted.shape[2]):
                if receiver_index % circumferential_elements == transmit_index:
                    continue
                modeled = scale * predicted[frequency_index, transmit_index, receiver_index]
                measured = observed[frequency_index, transmit_index, receiver_index]
                phase_errors.append(float(np.angle(modeled / measured)))
                log_amplitude_errors.append(float(np.log(abs(modeled) / abs(measured))))
                source_index = (receiver_index // circumferential_elements)
                source_index *= circumferential_elements
                source_index += transmit_index
                ranges.append(float(np.linalg.norm(elements[receiver_index] - elements[source_index])))
    return {
        "passive_pair_count": int(len(phase_errors)),
        "passive_range_min_m": float(np.min(ranges)),
        "passive_range_max_m": float(np.max(ranges)),
        "passive_phase_error_rms_rad": rms(phase_errors),
        "passive_phase_error_max_abs_rad": float(np.max(np.abs(phase_errors))),
        "passive_log_amplitude_error_rms": rms(log_amplitude_errors),
    }


def outgoing_green(
    source_xyz_m: np.ndarray,
    receiver_xyz_m: np.ndarray,
    wavenumber_rad_per_m: float,
    min_distance_m: float,
) -> np.ndarray:
    distance = np.linalg.norm(np.asarray(receiver_xyz_m) - source_xyz_m, axis=-1)
    distance = np.maximum(distance, min_distance_m)
    return np.exp(1j * wavenumber_rad_per_m * distance) / (4.0 * np.pi * distance)


def validated_elements(
    elements_xyz_m: np.ndarray,
    circumferential_elements: int,
    rows: int,
) -> np.ndarray:
    elements = np.asarray(elements_xyz_m, dtype=np.float64)
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("elements_xyz_m must have shape (element_count, 3)")
    if circumferential_elements < 2 or rows <= 0:
        raise ValueError("array topology must have at least two angular elements and one row")
    if elements.shape[0] != circumferential_elements * rows:
        raise ValueError("element count must equal circumferential_elements * rows")
    if not np.all(np.isfinite(elements)):
        raise ValueError("elements_xyz_m must contain finite coordinates")
    return elements


def rms(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("rms requires at least one value")
    return float(np.sqrt(np.mean(data * data)))
