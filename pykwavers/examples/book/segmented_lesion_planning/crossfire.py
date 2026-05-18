"""Crossfire aperture construction for Chapter 32 planning."""

from __future__ import annotations

import numpy as np

from .types import HybridPlanConfig, TransducerAperture


def select_crossfire_components(
    candidates: list[dict[str, object]],
    best: dict[str, object],
    config: HybridPlanConfig,
) -> list[TransducerAperture]:
    count = max(int(config.crossfire_aperture_count), 1)
    central = best["aperture"]
    selected = [central]
    central_angle = float(central.angle_deg)
    min_sep = float(config.crossfire_min_angle_separation_deg)
    lower = [
        item for item in candidates
        if float(item["aperture"].angle_deg) <= central_angle - min_sep
    ]
    upper = [
        item for item in candidates
        if float(item["aperture"].angle_deg) >= central_angle + min_sep
    ]
    for group in (lower, upper):
        if len(selected) >= count or not group:
            continue
        chosen = min(
            group,
            key=lambda item: (
                abs(float(item["aperture"].angle_deg) - central_angle),
                -float(item["metrics"]["score"]),
            ),
        )
        selected.append(chosen["aperture"])
    if len(selected) < count:
        remaining = [
            item for item in candidates
            if all(item["aperture"] is not existing for existing in selected)
        ]
        remaining.sort(key=lambda item: float(item["metrics"]["score"]), reverse=True)
        for item in remaining:
            angle = float(item["aperture"].angle_deg)
            if all(abs(angle - float(existing.angle_deg)) >= min_sep for existing in selected):
                selected.append(item["aperture"])
            if len(selected) >= count:
                break
    return selected[:count]


def build_composite_aperture(apertures: list[TransducerAperture]) -> TransducerAperture:
    element_positions = np.vstack([item.element_positions_m for item in apertures])
    element_inward = np.vstack([aperture_element_inward_units(item) for item in apertures])
    center = np.mean(np.vstack([item.center_m for item in apertures]), axis=0)
    inward = np.mean(element_inward, axis=0)
    inward /= max(float(np.linalg.norm(inward)), 1e-12)
    return TransducerAperture(
        angle_deg=float(apertures[0].angle_deg),
        center_m=center,
        inward_unit=inward,
        element_positions_m=element_positions,
        element_inward_units_m=element_inward,
        source_angles_deg=tuple(float(item.angle_deg) for item in apertures),
    )


def aperture_element_inward_units(aperture: TransducerAperture) -> np.ndarray:
    if aperture.element_inward_units_m is not None:
        return np.asarray(aperture.element_inward_units_m, dtype=float)
    return np.repeat(aperture.inward_unit[None, :], aperture.element_positions_m.shape[0], axis=0)
