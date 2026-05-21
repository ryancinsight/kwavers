"""Finite-grid PSTD Green diagnostics for Ali 2025 replication."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def source_kappa_filtered_observation_cube(
    elements_xyz_m: np.ndarray,
    circumferential_elements: int,
    rows: int,
    frequencies_hz: Sequence[float],
    sound_speed_m_s: float,
    spacing_m: float,
    grid_shape: Sequence[int],
    time_step_s: float,
) -> np.ndarray:
    elements = validated_elements(elements_xyz_m, circumferential_elements, rows)
    shape = validated_grid_shape(grid_shape)
    validate_sampling(sound_speed_m_s, spacing_m, time_step_s)
    centers = grid_centers(shape, spacing_m)
    min_distance = 0.5 * spacing_m
    output = np.empty((len(frequencies_hz), circumferential_elements, elements.shape[0]), dtype=np.complex128)
    for transmit_index in range(circumferential_elements):
        source_indices = [
            point_to_grid_index(elements[row * circumferential_elements + transmit_index], shape, spacing_m)
            for row in range(rows)
        ]
        weights = source_kappa_filtered_source_weights(
            shape,
            spacing_m,
            sound_speed_m_s,
            time_step_s,
            source_indices,
        ).reshape(-1)
        for frequency_index, frequency_hz in enumerate(frequencies_hz):
            if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
                raise ValueError("frequencies_hz entries must be positive and finite")
            wavenumber = 2.0 * np.pi * float(frequency_hz) / sound_speed_m_s
            for receiver_index, receiver in enumerate(elements):
                output[frequency_index, transmit_index, receiver_index] = np.dot(
                    weights,
                    outgoing_green(centers, receiver, wavenumber, min_distance),
                )
    return output


def source_kappa_filtered_source_weights(
    grid_shape: Sequence[int],
    spacing_m: float,
    sound_speed_m_s: float,
    time_step_s: float,
    source_indices: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    shape = validated_grid_shape(grid_shape)
    validate_sampling(sound_speed_m_s, spacing_m, time_step_s)
    mask = source_mask_from_indices(shape, source_indices)
    symbol = source_kappa_symbol(shape, spacing_m, sound_speed_m_s, time_step_s)
    return np.fft.ifftn(np.fft.fftn(mask) * symbol).real


def pstd_periodic_observation_cube(
    elements_xyz_m: np.ndarray,
    circumferential_elements: int,
    rows: int,
    frequencies_hz: Sequence[float],
    sound_speed_m_s: float,
    spacing_m: float,
    grid_shape: Sequence[int],
    time_step_s: float,
    time_steps_per_frequency: Sequence[int],
    frequency_bin_start_steps_per_frequency: Sequence[int],
    source_amplitude_pa: float,
) -> np.ndarray:
    elements = validated_elements(elements_xyz_m, circumferential_elements, rows)
    shape = validated_grid_shape(grid_shape)
    receiver_indices = [point_to_grid_index(point, shape, spacing_m) for point in elements]
    output = np.empty((len(frequencies_hz), circumferential_elements, elements.shape[0]), dtype=np.complex128)
    for frequency_index, frequency_hz in enumerate(frequencies_hz):
        steps = int(time_steps_per_frequency[frequency_index])
        bin_start = int(frequency_bin_start_steps_per_frequency[frequency_index])
        for transmit_index in range(circumferential_elements):
            source_indices = [
                point_to_grid_index(
                    elements[row * circumferential_elements + transmit_index],
                    shape,
                    spacing_m,
                )
                for row in range(rows)
            ]
            traces = pstd_periodic_pressure_traces(
                shape,
                spacing_m,
                sound_speed_m_s,
                time_step_s,
                float(frequency_hz),
                steps,
                source_indices,
                receiver_indices,
                source_amplitude_pa,
            )
            output[frequency_index, transmit_index, :] = frequency_bin_complex_traces(
                traces,
                float(frequency_hz),
                time_step_s,
                bin_start,
            )
    return output


def pstd_periodic_pressure_traces(
    grid_shape: Sequence[int],
    spacing_m: float,
    sound_speed_m_s: float,
    time_step_s: float,
    frequency_hz: float,
    steps: int,
    source_indices: Sequence[tuple[int, int, int]],
    receiver_indices: Sequence[tuple[int, int, int]],
    source_amplitude_pa: float,
) -> np.ndarray:
    shape = validated_grid_shape(grid_shape)
    validate_sampling(sound_speed_m_s, spacing_m, time_step_s)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not np.isfinite(source_amplitude_pa):
        raise ValueError("source_amplitude_pa must be finite")
    source_hat = np.fft.fftn(source_mask_from_indices(shape, source_indices))
    source_hat *= source_kappa_symbol(shape, spacing_m, sound_speed_m_s, time_step_s)
    theta_squared = pstd_leapfrog_theta_squared(shape, spacing_m, sound_speed_m_s, time_step_s)
    receiver_indices = validated_indices(receiver_indices, shape)
    p_previous = np.zeros(shape, dtype=np.complex128)
    p_current = np.zeros(shape, dtype=np.complex128)
    previous_signal = 0.0
    traces = np.empty((len(receiver_indices), steps), dtype=np.float64)
    angular_step = 2.0 * np.pi * frequency_hz * time_step_s
    gain = 2.0 * sound_speed_m_s * time_step_s * source_amplitude_pa / spacing_m
    for step in range(steps):
        signal = np.sin(angular_step * step)
        p_next = (2.0 - theta_squared) * p_current - p_previous
        p_next += gain * (signal - previous_signal) * source_hat
        field = np.fft.ifftn(p_next).real
        for receiver, index in enumerate(receiver_indices):
            traces[receiver, step] = field[index]
        p_previous = p_current
        p_current = p_next
        previous_signal = float(signal)
    return traces


def frequency_bin_complex_traces(
    traces: np.ndarray,
    frequency_hz: float,
    time_step_s: float,
    start_sample: int,
) -> np.ndarray:
    data = np.asarray(traces, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] <= start_sample:
        raise ValueError("traces must have shape (receiver, steps) with a valid start_sample")
    sample_indices = np.arange(start_sample, data.shape[1], dtype=np.float64)
    phase = -2.0 * np.pi * frequency_hz * sample_indices * time_step_s
    weights = np.cos(phase) + 1j * np.sin(phase)
    return (2.0 / sample_indices.size) * (data[:, start_sample:] @ weights)


def pstd_leapfrog_theta_squared(
    grid_shape: Sequence[int],
    spacing_m: float,
    sound_speed_m_s: float,
    time_step_s: float,
) -> np.ndarray:
    k_magnitude = wavenumber_magnitude(grid_shape, spacing_m)
    x = 0.5 * sound_speed_m_s * time_step_s * k_magnitude
    kappa = np.ones_like(k_magnitude)
    nonzero = np.abs(x) >= 1.0e-10
    kappa[nonzero] = np.sin(x[nonzero]) / x[nonzero]
    return np.square(sound_speed_m_s * time_step_s * k_magnitude * kappa)


def source_kappa_symbol(
    grid_shape: Sequence[int],
    spacing_m: float,
    sound_speed_m_s: float,
    time_step_s: float,
) -> np.ndarray:
    shape = validated_grid_shape(grid_shape)
    validate_sampling(sound_speed_m_s, spacing_m, time_step_s)
    return np.cos(0.5 * sound_speed_m_s * time_step_s * wavenumber_magnitude(shape, spacing_m))


def outgoing_green(
    source_xyz_m: np.ndarray,
    receiver_xyz_m: np.ndarray,
    wavenumber_rad_per_m: float,
    min_distance_m: float,
) -> np.ndarray:
    distance = np.linalg.norm(np.asarray(receiver_xyz_m) - source_xyz_m, axis=-1)
    distance = np.maximum(distance, min_distance_m)
    return np.exp(1j * wavenumber_rad_per_m * distance) / (4.0 * np.pi * distance)


def grid_centers(grid_shape: Sequence[int], spacing_m: float) -> np.ndarray:
    shape = validated_grid_shape(grid_shape)
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    axes = [(np.arange(axis, dtype=np.float64) - 0.5 * (axis - 1)) * spacing_m for axis in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3)


def wavenumber_magnitude(grid_shape: Sequence[int], spacing_m: float) -> np.ndarray:
    shape = validated_grid_shape(grid_shape)
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    axes = [2.0 * np.pi * np.fft.fftfreq(axis, d=spacing_m) for axis in shape]
    kx, ky, kz = np.meshgrid(*axes, indexing="ij")
    return np.sqrt(kx * kx + ky * ky + kz * kz)


def point_to_grid_index(
    point_xyz_m: np.ndarray,
    grid_shape: Sequence[int],
    spacing_m: float,
) -> tuple[int, int, int]:
    shape = validated_grid_shape(grid_shape)
    point = np.asarray(point_xyz_m, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_xyz_m must be a finite 3-vector")
    center = 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0) * spacing_m
    index = np.rint((center + point) / spacing_m).astype(int)
    if np.any(index < 0) or np.any(index >= np.asarray(shape)):
        raise ValueError(f"point {point.tolist()} maps outside grid shape {shape}")
    return tuple(int(axis) for axis in index)


def validated_grid_shape(grid_shape: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(int(axis) for axis in grid_shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError(f"grid_shape must contain three positive dimensions, got {grid_shape}")
    return shape


def source_mask_from_indices(
    grid_shape: Sequence[int],
    source_indices: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    shape = validated_grid_shape(grid_shape)
    mask = np.zeros(shape, dtype=np.float64)
    for index in validated_indices(source_indices, shape):
        mask[index] += 1.0
    if not np.any(mask):
        raise ValueError("at least one source index is required")
    return mask


def validated_indices(
    indices: Sequence[tuple[int, int, int]],
    grid_shape: Sequence[int],
) -> list[tuple[int, int, int]]:
    shape = validated_grid_shape(grid_shape)
    validated = []
    for index in indices:
        if len(index) != 3:
            raise ValueError("indices must be 3-D")
        i, j, k = (int(axis) for axis in index)
        if i < 0 or j < 0 or k < 0 or i >= shape[0] or j >= shape[1] or k >= shape[2]:
            raise ValueError(f"index {index} lies outside grid shape {shape}")
        validated.append((i, j, k))
    return validated


def validate_sampling(sound_speed_m_s: float, spacing_m: float, time_step_s: float) -> None:
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be positive and finite")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive and finite")


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
