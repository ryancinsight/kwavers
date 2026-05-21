"""PyO3 wrappers for Rust-owned Ali 2025 source-excitation diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def source_excitation_diagnostics(
    predicted: np.ndarray,
    observed: np.ndarray,
    frequencies_hz: Sequence[float],
    source_amplitude_pa: float,
    time_step_s: float,
    time_steps_per_frequency: Sequence[int],
    frequency_bin_start_steps_per_frequency: Sequence[int],
) -> dict[str, float | int | list[dict[str, float | int]]]:
    return dict(
        _kw().breast_fwi_source_excitation_diagnostics(
            np.asarray(predicted, dtype=np.complex128),
            np.asarray(observed, dtype=np.complex128),
            [float(value) for value in frequencies_hz],
            float(source_amplitude_pa),
            float(time_step_s),
            [int(value) for value in time_steps_per_frequency],
            [int(value) for value in frequency_bin_start_steps_per_frequency],
        )
    )


def sine_frequency_bin_coefficient(
    frequency_hz: float,
    time_step_s: float,
    total_steps: int,
    start_sample: int,
) -> complex:
    real, imag = _kw().breast_fwi_sine_frequency_bin_coefficient(
        float(frequency_hz),
        float(time_step_s),
        int(total_steps),
        int(start_sample),
    )
    return complex(real, imag)


def _kw():
    import pykwavers as kw

    return kw
