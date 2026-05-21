"""Acquisition rank bounds for reduced Ali 2025 FWI probes."""

from __future__ import annotations

import math
from enum import Enum
from typing import Sequence


class SourceScalingPolicy(str, Enum):
    FIXED = "fixed"
    ESTIMATED = "estimated"


def acquisition_identifiability(
    shape: tuple[int, int, int],
    frequencies_hz: Sequence[float],
    transmissions: int,
    receivers: int,
    source_scaling_policy: SourceScalingPolicy,
) -> dict[str, int | float | bool | str]:
    validate_shape(shape)
    validate_counts(frequencies_hz, transmissions, receivers)
    unknowns = math.prod(shape)
    frequency_count = len(frequencies_hz)
    complex_observations = frequency_count * transmissions * receivers
    real_observation_dof = 2 * complex_observations
    nuisance_dof = source_scale_real_dof(frequency_count, transmissions, source_scaling_policy)
    informative_dof = max(0, real_observation_dof - nuisance_dof)
    return {
        "unknown_voxels": unknowns,
        "frequency_count": frequency_count,
        "complex_observations": complex_observations,
        "real_observation_dof": real_observation_dof,
        "source_scaling_policy": source_scaling_policy.value,
        "estimated_source_scale_real_dof": nuisance_dof,
        "informative_real_dof_upper_bound": informative_dof,
        "informative_dof_to_unknown_ratio": informative_dof / unknowns,
        "underdetermined_by_rank_upper_bound": informative_dof < unknowns,
    }


def require_determined_acquisition(report: dict[str, int | float | bool | str]) -> None:
    if not report["underdetermined_by_rank_upper_bound"]:
        return
    raise ValueError(
        "acquisition is rank-underdetermined by upper bound: "
        f"{report['informative_real_dof_upper_bound']} informative real DoF "
        f"for {report['unknown_voxels']} unknown voxels"
    )


def source_scale_real_dof(
    frequency_count: int,
    transmissions: int,
    source_scaling_policy: SourceScalingPolicy,
) -> int:
    if source_scaling_policy == SourceScalingPolicy.ESTIMATED:
        return 2 * frequency_count * transmissions
    if source_scaling_policy == SourceScalingPolicy.FIXED:
        return 0
    raise ValueError(f"unsupported source scaling policy: {source_scaling_policy}")


def validate_shape(shape: tuple[int, int, int]) -> None:
    if len(shape) != 3 or any(axis <= 0 for axis in shape):
        raise ValueError(f"shape must contain three positive axes, got {shape}")


def validate_counts(
    frequencies_hz: Sequence[float],
    transmissions: int,
    receivers: int,
) -> None:
    if not frequencies_hz:
        raise ValueError("frequencies_hz must not be empty")
    for frequency_hz in frequencies_hz:
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError(f"frequency must be positive and finite, got {frequency_hz}")
    if transmissions <= 0:
        raise ValueError(f"transmissions must be positive, got {transmissions}")
    if receivers <= 0:
        raise ValueError(f"receivers must be positive, got {receivers}")
