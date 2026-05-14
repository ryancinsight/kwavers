"""HistoSonics-like therapy and imaging aperture geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransducerLayout:
    """Projected 2-D therapy-head geometry used by the Born FWI operator."""

    therapy_sources_m: np.ndarray
    therapy_receivers_m: np.ndarray
    imaging_receivers_m: np.ndarray
    metadata: dict[str, float | int | str]


def histosonics_like_layout(
    *,
    element_count: int,
    imaging_receiver_count: int,
    focal_radius_m: float,
    lateral_extent_m: float,
    central_cutout_m: float,
) -> TransducerLayout:
    """Return a public-geometry analog of an Edison-style treatment head.

    The layout is a projected 2-D analog, not a proprietary HistoSonics design.
    It follows public descriptions of a concave histotripsy therapy head with a
    central coaxial diagnostic ultrasound probe.
    """

    half_angle = float(np.arcsin(min(lateral_extent_m / (2.0 * focal_radius_m), 0.98)))
    cutout_angle = float(
        np.arcsin(min(central_cutout_m / (2.0 * focal_radius_m), 0.95))
    )
    lobes = split_arc_angles(element_count, half_angle, cutout_angle)
    sources = arc_points(lobes, focal_radius_m)
    probe = np.linspace(
        -0.5 * central_cutout_m,
        0.5 * central_cutout_m,
        imaging_receiver_count,
        dtype=np.float32,
    )
    imaging = np.column_stack(
        [probe, np.full(imaging_receiver_count, -focal_radius_m, dtype=np.float32)]
    )
    return TransducerLayout(
        therapy_sources_m=sources,
        therapy_receivers_m=sources,
        imaging_receivers_m=imaging.astype(np.float32),
        metadata={
            "model": "histosonics_like_projected_2d",
            "therapy_elements": int(element_count),
            "imaging_receivers": int(imaging_receiver_count),
            "focal_radius_m": float(focal_radius_m),
            "lateral_extent_m": float(lateral_extent_m),
            "central_cutout_m": float(central_cutout_m),
            "source": "public analog, not proprietary Edison geometry",
        },
    )


def split_arc_angles(count: int, half_angle: float, cutout_angle: float) -> np.ndarray:
    """Distribute therapy elements across the two aperture lobes."""

    left_count = count // 2
    right_count = count - left_count
    left = np.linspace(-half_angle, -cutout_angle, left_count, endpoint=True)
    right = np.linspace(cutout_angle, half_angle, right_count, endpoint=True)
    return np.concatenate([left, right]).astype(np.float32)


def arc_points(theta: np.ndarray, radius_m: float) -> np.ndarray:
    """Project concave therapy elements onto the inversion plane."""

    y = radius_m * np.sin(theta)
    x = -radius_m * np.cos(theta)
    return np.column_stack([y, x]).astype(np.float32)


def receiver_points_for_source(
    layout: TransducerLayout,
    source_index: int,
    therapy_offsets: tuple[int, ...],
    imaging_samples: int,
) -> list[np.ndarray]:
    """Return hybrid therapy-head and coaxial-imaging receiver points."""

    receivers = [
        layout.therapy_receivers_m[
            (source_index + offset) % layout.therapy_receivers_m.shape[0]
        ]
        for offset in therapy_offsets
    ]
    if imaging_samples > 0:
        stride = max(layout.imaging_receivers_m.shape[0] // imaging_samples, 1)
        start = source_index % stride
        indices = (
            start + stride * np.arange(imaging_samples)
        ) % layout.imaging_receivers_m.shape[0]
        receivers.extend(layout.imaging_receivers_m[indices])
    return receivers
