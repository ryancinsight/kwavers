"""Source-excitation diagnostics for Ali 2025 PSTD-vs-Helmholtz parity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .forward_consistency import complex_row_scale, paired_observation_cubes


def source_excitation_diagnostics(
    predicted: np.ndarray,
    observed: np.ndarray,
    frequencies_hz: Sequence[float],
    source_amplitude_pa: float,
    time_step_s: float,
    time_steps_per_frequency: Sequence[int],
    frequency_bin_start_steps_per_frequency: Sequence[int],
) -> dict[str, float | int | list[dict[str, float | int]]]:
    predicted, observed = paired_observation_cubes(predicted, observed)
    frequencies = finite_positive_sequence(frequencies_hz, "frequencies_hz")
    total_steps = positive_int_sequence(time_steps_per_frequency, "time_steps_per_frequency")
    bin_starts = nonnegative_int_sequence(
        frequency_bin_start_steps_per_frequency,
        "frequency_bin_start_steps_per_frequency",
    )
    validate_frequency_metadata(predicted.shape[0], frequencies, total_steps, bin_starts)
    if not np.isfinite(source_amplitude_pa) or source_amplitude_pa <= 0.0:
        raise ValueError("source_amplitude_pa must be positive and finite")
    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive and finite")

    rows = []
    for frequency_index, frequency_hz in enumerate(frequencies):
        bin_coefficient = sine_frequency_bin_coefficient(
            frequency_hz,
            time_step_s,
            total_steps[frequency_index],
            bin_starts[frequency_index],
        )
        row_scales = np.asarray(
            [
                complex_row_scale(
                    predicted[frequency_index, transmit_index, :],
                    observed[frequency_index, transmit_index, :],
                )
                for transmit_index in range(predicted.shape[1])
            ],
            dtype=np.complex128,
        )
        normalized = row_scales / (source_amplitude_pa * bin_coefficient)
        magnitudes = np.abs(normalized)
        phases = np.angle(normalized)
        resultant = abs(np.mean(np.exp(1j * phases)))
        mean_magnitude = float(np.mean(magnitudes))
        rows.append(
            {
                "frequency_hz": float(frequency_hz),
                "tone_bin_magnitude": float(abs(bin_coefficient)),
                "tone_bin_phase_rad": float(np.angle(bin_coefficient)),
                "mean_source_scale_magnitude": mean_magnitude,
                "source_scale_magnitude_coefficient_of_variation": float(
                    np.std(magnitudes) / max(mean_magnitude, np.finfo(np.float64).eps)
                ),
                "source_scale_phase_circular_variance": float(1.0 - resultant),
                "source_scale_phase_span_rad": phase_span_rad(phases),
            }
        )

    return {
        "frequency_count": int(predicted.shape[0]),
        "transmission_count": int(predicted.shape[1]),
        "source_amplitude_pa": float(source_amplitude_pa),
        "max_source_scale_magnitude_coefficient_of_variation": max(
            float(row["source_scale_magnitude_coefficient_of_variation"]) for row in rows
        ),
        "max_source_scale_phase_circular_variance": max(
            float(row["source_scale_phase_circular_variance"]) for row in rows
        ),
        "max_source_scale_phase_span_rad": max(
            float(row["source_scale_phase_span_rad"]) for row in rows
        ),
        "per_frequency": rows,
    }


def sine_frequency_bin_coefficient(
    frequency_hz: float,
    time_step_s: float,
    total_steps: int,
    start_sample: int,
) -> complex:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive and finite")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if start_sample < 0 or start_sample >= total_steps:
        raise ValueError("start_sample must lie inside the trace")
    samples = np.arange(start_sample, total_steps, dtype=np.float64)
    phase = 2.0 * np.pi * frequency_hz * samples * time_step_s
    values = np.sin(phase) * np.exp(-1j * phase)
    coefficient = (2.0 / values.size) * np.sum(values)
    if abs(coefficient) <= np.finfo(np.float64).eps:
        raise ValueError("sine frequency-bin coefficient has zero energy")
    return complex(coefficient)


def phase_span_rad(phases: np.ndarray) -> float:
    if phases.size == 0:
        raise ValueError("phase vector must not be empty")
    return float(np.max(np.unwrap(phases)) - np.min(np.unwrap(phases)))


def validate_frequency_metadata(
    frequency_count: int,
    frequencies_hz: Sequence[float],
    time_steps_per_frequency: Sequence[int],
    frequency_bin_start_steps_per_frequency: Sequence[int],
) -> None:
    if len(frequencies_hz) != frequency_count:
        raise ValueError("frequencies_hz length must match observation frequency axis")
    if len(time_steps_per_frequency) != frequency_count:
        raise ValueError("time_steps_per_frequency length must match observation frequency axis")
    if len(frequency_bin_start_steps_per_frequency) != frequency_count:
        raise ValueError(
            "frequency_bin_start_steps_per_frequency length must match observation frequency axis"
        )
    for total_steps, start_step in zip(
        time_steps_per_frequency,
        frequency_bin_start_steps_per_frequency,
    ):
        if start_step >= total_steps:
            raise ValueError("frequency bin start must be smaller than total steps")


def finite_positive_sequence(values: Sequence[float], name: str) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if any((not np.isfinite(value)) or value <= 0.0 for value in parsed):
        raise ValueError(f"{name} entries must be positive and finite")
    return parsed


def positive_int_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if any(value <= 0 for value in parsed):
        raise ValueError(f"{name} entries must be positive")
    return parsed


def nonnegative_int_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if any(value < 0 for value in parsed):
        raise ValueError(f"{name} entries must be nonnegative")
    return parsed
