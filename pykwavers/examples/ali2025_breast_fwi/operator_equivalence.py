"""Forward-operator equivalence diagnostics for Ali 2025 replication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .forward_consistency import scaled_observation_residual_metrics
from .source_excitation import source_excitation_diagnostics

PROPAGATION_MODELS = (
    "single_scatter_born",
    "dense_convergent_born",
    "spectral_convergent_born",
)


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
) -> dict[str, Any]:
    if not predictions_by_model:
        raise ValueError("predictions_by_model must not be empty")

    per_model = []
    for model, predicted in predictions_by_model.items():
        residual = scaled_observation_residual_metrics(predicted, observed)
        excitation = source_excitation_diagnostics(
            predicted,
            observed,
            frequencies_hz,
            source_amplitude_pa,
            time_step_s,
            time_steps_per_frequency,
            frequency_bin_start_steps_per_frequency,
        )
        per_model.append(
            {
                "model": str(model),
                "normalized_l2_residual": float(residual["normalized_l2_residual"]),
                "row_normalized_l2_residual_mean": float(
                    residual["row_normalized_l2_residual_mean"]
                ),
                "source_scale_magnitude_coefficient_of_variation": float(
                    excitation["max_source_scale_magnitude_coefficient_of_variation"]
                ),
                "source_scale_phase_span_rad": float(
                    excitation["max_source_scale_phase_span_rad"]
                ),
            }
        )

    per_model.sort(key=lambda entry: entry["normalized_l2_residual"])
    best = per_model[0]
    worst = per_model[-1]
    return {
        "model_count": len(per_model),
        "best_model": best["model"],
        "best_normalized_l2_residual": best["normalized_l2_residual"],
        "worst_model": worst["model"],
        "worst_normalized_l2_residual": worst["normalized_l2_residual"],
        "residual_spread": worst["normalized_l2_residual"] - best["normalized_l2_residual"],
        "per_model": per_model,
    }
