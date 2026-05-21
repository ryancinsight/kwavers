"""PyO3-backed homogeneous direct-field diagnostics for Ali 2025 PSTD parity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def homogeneous_direct_field_probe(
    kw: Any,
    homogeneous_sound_speed_m_s: Any,
    array: Any,
    frequencies_hz: Sequence[float],
    dataset_config: Any,
) -> dict[str, Any]:
    return dict(
        kw.diagnose_breast_fwi_homogeneous_direct_field(
            homogeneous_sound_speed_m_s,
            array,
            list(frequencies_hz),
            dataset_config,
        )
    )
