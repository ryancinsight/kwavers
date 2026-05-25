"""Forward-operator equivalence diagnostics for Ali 2025 replication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

PROPAGATION_MODELS = (
    "single_scatter_born",
    "dense_convergent_born",
    "spectral_convergent_born",
    "pstd_spectral_convergent_born",
)


class ReceiverChannelPolicy(str, Enum):
    ALL = "all"
    ACTIVE_ONLY = "active_only"
    PASSIVE_ONLY = "passive_only"


def make_frequency_domain_fwi_config(
    kw: Any,
    args: Any,
    reference_sound_speed_m_s: float,
    spacing_m: float,
    propagation_model: str,
) -> Any:
    return kw.FrequencyDomainFwiConfig(
        reference_sound_speed_m_s=reference_sound_speed_m_s,
        spacing_m=spacing_m,
        iterations=args.fwi_iterations,
        initial_step_s_per_m=args.initial_step_s_per_m,
        min_sound_speed_m_s=args.min_sound_speed_m_s,
        max_sound_speed_m_s=args.max_sound_speed_m_s,
        estimate_source_scaling=True,
        tikhonov_weight=args.tikhonov_weight,
        propagation_model=propagation_model,
        cbs_iterations=args.cbs_iterations,
        cbs_relative_tolerance=args.cbs_relative_tolerance,
        absorbing_boundary="polynomial",
        absorbing_thickness_cells=args.absorbing_thickness_cells,
        absorbing_strength_nepers=args.absorbing_strength_nepers,
        absorbing_order=args.absorbing_order,
        pstd_time_step_s=args.time_step_s,
        pstd_source_amplitude_pa=args.source_amplitude_pa,
        pstd_cycles_per_frequency=args.cycles_per_frequency,
        pstd_frequency_bin_cycles=args.frequency_bin_cycles,
    )


def make_configs_by_model(
    kw: Any,
    args: Any,
    reference_sound_speed_m_s: float,
    spacing_m: float,
) -> dict[str, Any]:
    return {
        model: make_frequency_domain_fwi_config(
            kw,
            args,
            reference_sound_speed_m_s,
            spacing_m,
            model,
        )
        for model in PROPAGATION_MODELS
    }


def simulate_forward_predictions(
    kw: Any,
    sound_speed_m_s: np.ndarray,
    array: Any,
    frequencies_hz: Sequence[float],
    configs_by_model: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not configs_by_model:
        raise ValueError("configs_by_model must not be empty")
    return {
        model: np.stack(
            [
                kw.simulate_breast_fwi_frequency_observation(
                    sound_speed_m_s,
                    array,
                    frequency_hz,
                    config,
                )
                for frequency_hz in frequencies_hz
            ],
            axis=0,
        )
        for model, config in configs_by_model.items()
    }


def operator_equivalence_diagnostics(
    predictions_by_model: Mapping[str, np.ndarray],
    observed: np.ndarray,
    frequencies_hz: Sequence[float],
    source_amplitude_pa: float,
    time_step_s: float,
    time_steps_per_frequency: Sequence[int],
    frequency_bin_start_steps_per_frequency: Sequence[int],
    receiver_channel_policy: ReceiverChannelPolicy | str = ReceiverChannelPolicy.ALL,
) -> dict[str, Any]:
    predictions = {
        str(model): np.asarray(predicted, dtype=np.complex128)
        for model, predicted in predictions_by_model.items()
    }
    return dict(
        _kw().breast_fwi_operator_equivalence_diagnostics(
            predictions,
            np.asarray(observed, dtype=np.complex128),
            [float(value) for value in frequencies_hz],
            float(source_amplitude_pa),
            float(time_step_s),
            [int(value) for value in time_steps_per_frequency],
            [int(value) for value in frequency_bin_start_steps_per_frequency],
            ReceiverChannelPolicy(receiver_channel_policy).value,
        )
    )


def scattering_increment_diagnostics(
    homogeneous_baseline: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray],
    observed: np.ndarray,
    receiver_channel_policy: ReceiverChannelPolicy | str = ReceiverChannelPolicy.ALL,
) -> dict[str, Any]:
    predictions = {
        str(model): np.asarray(predicted, dtype=np.complex128)
        for model, predicted in predictions_by_model.items()
    }
    return dict(
        _kw().breast_fwi_scattering_increment_diagnostics(
            np.asarray(homogeneous_baseline, dtype=np.complex128),
            predictions,
            np.asarray(observed, dtype=np.complex128),
            ReceiverChannelPolicy(receiver_channel_policy).value,
        )
    )


def _kw() -> Any:
    import pykwavers as kw

    return kw
