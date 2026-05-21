"""PyO3 wrappers for Rust-owned Ali 2025 acquisition rank bounds."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum


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
    return dict(
        _kw().breast_fwi_acquisition_identifiability(
            _shape3(shape),
            [float(value) for value in frequencies_hz],
            int(transmissions),
            int(receivers),
            SourceScalingPolicy(source_scaling_policy).value,
        )
    )


def require_determined_acquisition(report: dict[str, int | float | bool | str]) -> None:
    if not report["underdetermined_by_rank_upper_bound"]:
        return
    raise ValueError(
        "acquisition is rank-underdetermined by upper bound: "
        f"{report['informative_real_dof_upper_bound']} informative real DoF "
        f"for {report['unknown_voxels']} unknown voxels"
    )


def _kw():
    import pykwavers as kw

    return kw


def _shape3(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"shape must contain three positive axes, got {shape}")
    return tuple(int(axis) for axis in shape)
