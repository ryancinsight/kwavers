from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import CANONICAL_BRAIN_SCENE, BrainSceneDefinition


@dataclass(frozen=True)
class TransducerConfig:
    element_count: int = CANONICAL_BRAIN_SCENE.transducer.element_count
    frequency_hz: float = CANONICAL_BRAIN_SCENE.transducer.frequency_hz
    radius_m: float = CANONICAL_BRAIN_SCENE.transducer.radius_m
    cap_min_polar_rad: float = CANONICAL_BRAIN_SCENE.transducer.cap_min_polar_rad
    cap_max_polar_rad: float = CANONICAL_BRAIN_SCENE.transducer.cap_max_polar_rad
    brain_sound_speed_m_s: float = CANONICAL_BRAIN_SCENE.transducer.brain_sound_speed_m_s
    skull_sound_speed_m_s: float = CANONICAL_BRAIN_SCENE.transducer.skull_sound_speed_m_s

    @classmethod
    def from_scene(cls, scene: BrainSceneDefinition = CANONICAL_BRAIN_SCENE) -> "TransducerConfig":
        return cls(**scene.transducer_config_kwargs())


@dataclass(frozen=True)
class PhaseCorrection:
    element_positions_m: np.ndarray
    phases_rad: np.ndarray
    delays_s: np.ndarray
    skull_lengths_m: np.ndarray
    amplitude_weights: np.ndarray
    active: np.ndarray


def fibonacci_hemisphere(config: TransducerConfig) -> np.ndarray:
    n = config.element_count
    idx = np.arange(n, dtype=np.float64) + 0.5
    golden = np.pi * (3.0 - np.sqrt(5.0))
    cos_min = np.cos(config.cap_min_polar_rad)
    cos_max = np.cos(config.cap_max_polar_rad)
    cos_theta = cos_min + (cos_max - cos_min) * idx / n
    theta = np.arccos(cos_theta)
    phi = golden * idx
    x = config.radius_m * np.sin(theta) * np.cos(phi)
    y = config.radius_m * np.sin(theta) * np.sin(phi)
    z = -config.radius_m * np.cos(theta)
    return np.column_stack([x, y, z]).astype(np.float64)


def phase_correction_through_skull(
    skull_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
    config: TransducerConfig,
    samples_per_ray: int = 192,
) -> PhaseCorrection:
    return phase_correction_through_ct(
        skull_mask.astype(np.float32) * 1000.0,
        spacing_m,
        target_index,
        config,
        samples_per_ray,
        skull_mask=skull_mask,
    )


def phase_correction_through_ct(
    ct_hu: np.ndarray,
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
    config: TransducerConfig,
    samples_per_ray: int = 192,
    skull_mask: np.ndarray | None = None,
) -> PhaseCorrection:
    positions = fibonacci_hemisphere(config)
    target = index_to_point(np.asarray(target_index), spacing_m, target_index)
    delays = np.zeros(config.element_count, dtype=np.float64)
    lengths = np.zeros(config.element_count, dtype=np.float64)
    amplitudes = np.ones(config.element_count, dtype=np.float64)
    active = np.ones(config.element_count, dtype=bool)
    mask = skull_mask if skull_mask is not None else ct_hu > 300.0

    for idx, element in enumerate(positions):
        length, delay, amplitude = skull_path_acoustics_from_ct(
            element,
            target,
            ct_hu,
            mask,
            spacing_m,
            target_index,
            config,
            samples_per_ray,
        )
        delays[idx] = delay
        lengths[idx] = length
        amplitudes[idx] = amplitude

    relative = delays - float(np.mean(delays[active]))
    phases = -2.0 * np.pi * config.frequency_hz * relative
    phases = np.angle(np.exp(1j * phases))
    return PhaseCorrection(positions, phases.astype(np.float64), delays, lengths, amplitudes, active)


def skull_path_acoustics_from_ct(
    start_m: np.ndarray,
    end_m: np.ndarray,
    ct_hu: np.ndarray,
    skull_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
    config: TransducerConfig,
    samples_per_ray: int,
) -> tuple[float, float, float]:
    t = np.linspace(0.0, 1.0, samples_per_ray, dtype=np.float64)
    points = start_m[None, :] * (1.0 - t[:, None]) + end_m[None, :] * t[:, None]
    indices = point_to_index(points, spacing_m, target_index)
    inside = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < ct_hu.shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < ct_hu.shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < ct_hu.shape[2])
    )
    segment_m = float(np.linalg.norm(end_m - start_m) / max(samples_per_ray - 1, 1))
    delay_s = 0.0
    skull_length_m = 0.0
    attenuation_np = 0.0
    transmission = 1.0
    prev_impedance = 1040.0 * config.brain_sound_speed_m_s

    for is_inside, ijk in zip(inside, indices):
        if not is_inside:
            sound_speed = config.brain_sound_speed_m_s
            impedance = 1040.0 * sound_speed
            alpha_np_m = 0.0
            in_skull = False
        else:
            hu = float(ct_hu[ijk[0], ijk[1], ijk[2]])
            in_skull = bool(skull_mask[ijk[0], ijk[1], ijk[2]])
            sound_speed, density, alpha_np_m = acoustic_properties_from_hu(
                hu,
                config.frequency_hz,
                config.brain_sound_speed_m_s,
                config.skull_sound_speed_m_s,
            )
            impedance = density * sound_speed
        if abs(impedance - prev_impedance) / max(prev_impedance, impedance) > 0.05:
            intensity_transmission = 4.0 * prev_impedance * impedance / ((prev_impedance + impedance) ** 2)
            transmission *= float(np.sqrt(np.clip(intensity_transmission, 0.0, 1.0)))
        delay_s += segment_m / sound_speed
        attenuation_np += alpha_np_m * segment_m
        skull_length_m += segment_m if in_skull else 0.0
        prev_impedance = impedance

    return skull_length_m, delay_s, float(transmission * np.exp(-attenuation_np))


def acoustic_properties_from_hu(
    hu: float,
    frequency_hz: float,
    brain_sound_speed_m_s: float,
    skull_sound_speed_m_s: float,
) -> tuple[float, float, float]:
    if hu <= 300.0:
        return brain_sound_speed_m_s, 1040.0, 0.5 * 100.0 / 8.686 * (frequency_hz / 1.0e6)
    bone_fraction = float(np.clip((hu - 300.0) / 1700.0, 0.0, 1.0))
    density = 1200.0 + 700.0 * bone_fraction
    sound_speed = brain_sound_speed_m_s + (skull_sound_speed_m_s - brain_sound_speed_m_s) * bone_fraction
    attenuation_db_cm_mhz = 8.0 + 12.0 * bone_fraction
    alpha_np_m = attenuation_db_cm_mhz * 100.0 / 8.686 * (frequency_hz / 1.0e6)
    return sound_speed, density, alpha_np_m


def skull_intersection_length(
    start_m: np.ndarray,
    end_m: np.ndarray,
    skull_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
    samples_per_ray: int,
) -> float:
    t = np.linspace(0.0, 1.0, samples_per_ray, dtype=np.float64)
    points = start_m[None, :] * (1.0 - t[:, None]) + end_m[None, :] * t[:, None]
    indices = point_to_index(points, spacing_m, target_index)
    inside = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < skull_mask.shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < skull_mask.shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < skull_mask.shape[2])
    )
    hit = np.zeros(samples_per_ray, dtype=bool)
    valid = indices[inside]
    hit[inside] = skull_mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    segment = float(np.linalg.norm(end_m - start_m) / max(samples_per_ray - 1, 1))
    return float(np.count_nonzero(hit) * segment)


def index_to_point(
    index: np.ndarray,
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
) -> np.ndarray:
    return (index.astype(np.float64) - np.asarray(target_index, dtype=np.float64)) * np.asarray(spacing_m)


def point_to_index(
    points_m: np.ndarray,
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
) -> np.ndarray:
    index = points_m / np.asarray(spacing_m) + np.asarray(target_index, dtype=np.float64)
    return np.rint(index).astype(int)
