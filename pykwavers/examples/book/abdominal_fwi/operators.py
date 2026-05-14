"""Finite-frequency channel operators for abdominal FWI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abdominal_fwi.constants import C_REF_M_S
from abdominal_fwi.transducer import histosonics_like_layout, receiver_points_for_source


@dataclass(frozen=True)
class ChannelSpec:
    """Compile-time-style constants for one deterministic FWI channel."""

    name: str
    source_phase_multiplier: float
    receiver_phase_multiplier: float
    source_attenuation_multiplier: float
    receiver_attenuation_multiplier: float
    source_distance_power: float
    receiver_distance_power: float
    use_nonlinear_weight: bool = False
    use_path_sum_phase: bool = False


FUNDAMENTAL = ChannelSpec(
    name="fundamental",
    source_phase_multiplier=1.0,
    receiver_phase_multiplier=1.0,
    source_attenuation_multiplier=1.0,
    receiver_attenuation_multiplier=1.0,
    source_distance_power=0.5,
    receiver_distance_power=0.5,
    use_path_sum_phase=True,
)

SUBHARMONIC = ChannelSpec(
    name="subharmonic",
    source_phase_multiplier=1.0,
    receiver_phase_multiplier=0.5,
    source_attenuation_multiplier=1.0,
    receiver_attenuation_multiplier=0.5,
    source_distance_power=0.5,
    receiver_distance_power=0.5,
)

NONLINEAR = ChannelSpec(
    name="nonlinear_harmonic",
    source_phase_multiplier=2.0,
    receiver_phase_multiplier=2.0,
    source_attenuation_multiplier=2.0,
    receiver_attenuation_multiplier=2.0,
    source_distance_power=1.0,
    receiver_distance_power=0.5,
    use_nonlinear_weight=True,
)


def build_fundamental_matrix(prepared, active_mask: np.ndarray, config) -> np.ndarray:
    """Assemble the linear finite-frequency Born matrix."""

    return build_source_receiver_matrix(prepared, active_mask, config, FUNDAMENTAL)


def build_subharmonic_matrix(prepared, active_mask: np.ndarray, config) -> np.ndarray:
    """Assemble source-conditioned subharmonic cavitation-emission sensitivity."""

    return build_source_receiver_matrix(prepared, active_mask, config, SUBHARMONIC)


def build_nonlinear_matrix(prepared, active_mask: np.ndarray, config) -> np.ndarray:
    """Assemble weak-Westervelt second-harmonic sensitivity."""

    return build_source_receiver_matrix(prepared, active_mask, config, NONLINEAR)


def build_source_receiver_matrix(
    prepared,
    active_mask: np.ndarray,
    config,
    channel: ChannelSpec,
) -> np.ndarray:
    """Assemble a normalized source/receiver sensitivity matrix."""

    points = active_points(active_mask, prepared.spacing_m)
    layout = histosonics_like_layout(
        element_count=config.element_count,
        imaging_receiver_count=config.imaging_receiver_count,
        focal_radius_m=config.therapy_focal_radius_m,
        lateral_extent_m=config.therapy_lateral_extent_m,
        central_cutout_m=config.therapy_central_cutout_m,
    )
    attenuation = attenuation_map(prepared)
    material = nonlinear_weight(prepared, active_mask) if channel.use_nonlinear_weight else 1.0
    area = prepared.spacing_m * prepared.spacing_m
    rows = []

    for src_idx, source in enumerate(layout.therapy_sources_m):
        ds = np.linalg.norm(points - source[None, :], axis=1) + prepared.spacing_m
        source_loss = path_attenuation(attenuation, points, source, prepared.spacing_m)
        receivers = receiver_points_for_source(
            layout,
            src_idx,
            config.receiver_offsets,
            config.imaging_receiver_samples,
        )
        for receiver in receivers:
            dr = np.linalg.norm(points - receiver[None, :], axis=1) + prepared.spacing_m
            receiver_loss = path_loss(attenuation, points, receiver, prepared.spacing_m)
            rows.extend(
                channel_rows(
                    config.frequencies_hz,
                    ds,
                    dr,
                    source_loss,
                    receiver_loss,
                    area,
                    material,
                    channel,
                )
            )

    if not rows:
        raise ValueError(f"{prepared.name}: no nonzero {channel.name} rows assembled")
    return np.vstack(rows).astype(np.float32)


def channel_rows(
    frequencies_hz: tuple[float, ...],
    ds: np.ndarray,
    dr: np.ndarray,
    source_loss: np.ndarray,
    receiver_loss: np.ndarray,
    area: float,
    material: np.ndarray | float,
    channel: ChannelSpec,
) -> list[np.ndarray]:
    """Build normalized rows for one source/receiver pair."""

    rows = []
    for frequency_hz in frequencies_hz:
        k = 2.0 * np.pi * frequency_hz / C_REF_M_S
        mhz = frequency_hz * 1.0e-6
        damping = np.exp(
            -(
                source_loss * channel.source_attenuation_multiplier
                + receiver_loss * channel.receiver_attenuation_multiplier
            )
            * mhz
        )
        if channel.use_path_sum_phase:
            distance = np.power(ds, channel.source_distance_power) * np.power(
                dr,
                channel.receiver_distance_power,
            )
            phase = np.cos(k * (ds + dr))
            row = area * material * damping * phase / distance
        else:
            source_wave = np.cos(channel.source_phase_multiplier * k * ds) / np.power(
                ds,
                channel.source_distance_power,
            )
            receiver_wave = np.cos(channel.receiver_phase_multiplier * k * dr) / np.power(
                dr,
                channel.receiver_distance_power,
            )
            row = area * material * damping * source_wave * receiver_wave
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            rows.append((row / norm).astype(np.float32))
    return rows


def active_points(mask: np.ndarray, spacing_m: float) -> np.ndarray:
    """Return active voxel centers in meters with the slice center as origin."""

    yy, xx = np.where(mask)
    center = (np.asarray(mask.shape, dtype=np.float32) - 1.0) * 0.5
    y = (yy.astype(np.float32) - center[0]) * spacing_m
    x = (xx.astype(np.float32) - center[1]) * spacing_m
    return np.column_stack([y, x]).astype(np.float32)


def attenuation_map(prepared) -> np.ndarray:
    """Map CT HU and labels to path attenuation in Np/(m MHz)."""

    ct = prepared.ct_hu.astype(np.float32)
    alpha = np.full(ct.shape, 0.7, dtype=np.float32)
    alpha[ct < -500.0] = 0.05
    alpha[(ct >= -500.0) & (ct < -50.0)] = 0.35
    alpha[ct > 250.0] = 18.0
    alpha[prepared.imaging_mask & (ct > -50.0) & (ct <= 250.0)] = 0.65
    alpha[prepared.organ_mask] = 0.8
    alpha[prepared.target_mask] = 1.0
    return alpha


def nonlinear_weight(prepared, active_mask: np.ndarray) -> np.ndarray:
    """Return bounded CT-derived nonlinear scattering weights."""

    speed = prepared.sound_speed_m_s.astype(np.float32)
    active_speed = speed[active_mask]
    span = max(float(np.ptp(active_speed)), 1.0)
    return (1.0 + np.abs(active_speed - float(np.median(active_speed))) / span).astype(
        np.float32
    )


def path_attenuation(
    attenuation: np.ndarray,
    points: np.ndarray,
    element: np.ndarray,
    spacing_m: float,
    sample_count: int = 10,
) -> np.ndarray:
    """Integrate attenuation along straight source/receiver voxel chords."""

    t = np.linspace(0.08, 0.92, sample_count, dtype=np.float32)
    samples = (
        element[None, None, :] * (1.0 - t[:, None, None])
        + points[None, :, :] * t[:, None, None]
    )
    values = bilinear(attenuation, samples.reshape(-1, 2), spacing_m)
    length = np.linalg.norm(points - element[None, :], axis=1)
    return values.reshape(sample_count, -1).mean(axis=0) * length


def path_loss(
    attenuation: np.ndarray,
    points: np.ndarray,
    receiver: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    """Receiver-side attenuation helper for a single receiver element."""

    return path_attenuation(attenuation, points, receiver, spacing_m)


def bilinear(image: np.ndarray, points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Sample a centered 2-D image at physical coordinates in meters."""

    center = (np.asarray(image.shape, dtype=np.float32) - 1.0) * 0.5
    y = points[:, 0] / spacing_m + center[0]
    x = points[:, 1] / spacing_m + center[1]
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    valid = (y0 >= 0) & (x0 >= 0) & (y1 < image.shape[0]) & (x1 < image.shape[1])
    out = np.full(points.shape[0], float(np.max(image)), dtype=np.float32)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    out[valid] = (
        (1.0 - wy[valid]) * (1.0 - wx[valid]) * image[y0[valid], x0[valid]]
        + wy[valid] * (1.0 - wx[valid]) * image[y1[valid], x0[valid]]
        + (1.0 - wy[valid]) * wx[valid] * image[y0[valid], x1[valid]]
        + wy[valid] * wx[valid] * image[y1[valid], x1[valid]]
    )
    return out
