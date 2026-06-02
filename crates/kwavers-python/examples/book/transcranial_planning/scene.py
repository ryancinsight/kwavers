from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CtAlignedTargetSpec:
    name: str
    fraction_xyz: tuple[float, float, float]
    source_world_mm: tuple[float, float, float]
    source_frame: str

    def resolve_index(self, support_mask: np.ndarray) -> tuple[int, int, int]:
        return tuple(int(v) for v in _fraction_index(np.asarray(support_mask, dtype=bool), self.fraction_xyz))

    def resolve_index_2d(self, support_mask: np.ndarray) -> tuple[int, int]:
        index = _fraction_index(np.asarray(support_mask, dtype=bool), self.fraction_xyz[:2])
        return int(index[0]), int(index[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_fraction_xyz": list(self.fraction_xyz),
            "source_world_mm": list(self.source_world_mm),
            "source_frame": self.source_frame,
        }


@dataclass(frozen=True)
class TransducerPoseSpec:
    element_count: int
    frequency_hz: float
    radius_m: float
    cap_min_polar_rad: float
    cap_max_polar_rad: float
    aperture_diameter_m: float
    brain_sound_speed_m_s: float
    skull_sound_speed_m_s: float
    target_peak_pressure_pa: float
    diagnostic_source_pressure_pa: float
    samples_per_ray: int
    skull_hu_threshold: float
    body_hu_threshold: float

    def transducer_kwargs(self) -> dict[str, Any]:
        return {
            "element_count": self.element_count,
            "frequency_hz": self.frequency_hz,
            "radius_m": self.radius_m,
            "cap_min_polar_rad": self.cap_min_polar_rad,
            "cap_max_polar_rad": self.cap_max_polar_rad,
            "brain_sound_speed_m_s": self.brain_sound_speed_m_s,
            "skull_sound_speed_m_s": self.skull_sound_speed_m_s,
        }

    def pykwavers_kwargs(self) -> dict[str, Any]:
        return {
            "element_count": self.element_count,
            "frequency_hz": self.frequency_hz,
            "radius_m": self.radius_m,
            "cap_min_polar_rad": self.cap_min_polar_rad,
            "cap_max_polar_rad": self.cap_max_polar_rad,
            "brain_sound_speed": self.brain_sound_speed_m_s,
            "skull_sound_speed": self.skull_sound_speed_m_s,
            "target_peak_pa": self.target_peak_pressure_pa,
            "samples_per_ray": self.samples_per_ray,
            "skull_hu_threshold": self.skull_hu_threshold,
            "body_hu_threshold": self.body_hu_threshold,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_count": self.element_count,
            "frequency_hz": self.frequency_hz,
            "radius_m": self.radius_m,
            "cap_min_polar_rad": self.cap_min_polar_rad,
            "cap_max_polar_rad": self.cap_max_polar_rad,
            "aperture_diameter_m": self.aperture_diameter_m,
            "brain_sound_speed_m_s": self.brain_sound_speed_m_s,
            "skull_sound_speed_m_s": self.skull_sound_speed_m_s,
            "target_peak_pressure_pa": self.target_peak_pressure_pa,
            "diagnostic_source_pressure_pa": self.diagnostic_source_pressure_pa,
            "samples_per_ray": self.samples_per_ray,
            "skull_hu_threshold": self.skull_hu_threshold,
            "body_hu_threshold": self.body_hu_threshold,
        }


@dataclass(frozen=True)
class BrainSceneDefinition:
    name: str
    target: CtAlignedTargetSpec
    transducer: TransducerPoseSpec

    def transducer_config_kwargs(self) -> dict[str, Any]:
        return self.transducer.transducer_kwargs()

    def fus_pykwavers_kwargs(self) -> dict[str, Any]:
        kwargs = self.transducer.pykwavers_kwargs()
        kwargs["target_fraction_xyz"] = self.target.fraction_xyz
        return kwargs

    def benchmark_pykwavers_kwargs(self) -> dict[str, Any]:
        kwargs = self.fus_pykwavers_kwargs()
        kwargs["aperture_diameter_m"] = self.transducer.aperture_diameter_m
        return kwargs

    def inverse_pykwavers_kwargs(self) -> dict[str, Any]:
        return {
            "target_fraction_xyz": self.target.fraction_xyz,
            "element_count": self.transducer.element_count,
            "source_pressure_pa": self.transducer.diagnostic_source_pressure_pa,
        }

    def nonlinear_pykwavers_kwargs(self) -> dict[str, Any]:
        return {
            "target_fraction_xyz": self.target.fraction_xyz,
            "element_count": self.transducer.element_count,
            "frequency_hz": self.transducer.frequency_hz,
            "source_pressure_pa": self.transducer.diagnostic_source_pressure_pa,
        }

    def focused_bowl_pykwavers_kwargs(self) -> dict[str, Any]:
        return {
            "target_fraction_xyz": self.target.fraction_xyz,
            "element_count": self.transducer.element_count,
            "body_hu_threshold": self.transducer.body_hu_threshold,
            "skull_hu_threshold": self.transducer.skull_hu_threshold,
            "scene_radius_m": self.transducer.radius_m,
            "cap_min_polar_rad": self.transducer.cap_min_polar_rad,
            "cap_max_polar_rad": self.transducer.cap_max_polar_rad,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target.to_dict(),
            "transducer": self.transducer.to_dict(),
            "invariant": (
                "target_fraction_xyz resolves against the CT-derived brain support "
                "on each simulation lattice; transducer coordinates are generated "
                "from this single pose block and expressed relative to that target."
            ),
        }


def _fraction_index(mask: np.ndarray, fraction: tuple[float, ...]) -> np.ndarray:
    if mask.ndim != len(fraction):
        raise ValueError("target fraction dimensionality must match support mask")
    if any((not np.isfinite(value)) or value < 0.0 or value > 1.0 for value in fraction):
        raise ValueError("target fractions must be finite values in [0, 1]")
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("target support mask is empty")
    lo = coords.min(axis=0).astype(np.float64)
    hi = coords.max(axis=0).astype(np.float64)
    resolved = np.floor(lo + np.asarray(fraction, dtype=np.float64) * (hi - lo) + 0.5)
    return np.clip(resolved, lo, hi).astype(int)


CANONICAL_BRAIN_SCENE = BrainSceneDefinition(
    name="rire_ct_aligned_vim_like_scene",
    target=CtAlignedTargetSpec(
        name="vim_like_atlas_coordinate_resolved_to_ct_brain_support",
        fraction_xyz=(0.5925925925925926, 0.5, 0.4888888888888889),
        source_world_mm=(14.0, -18.0, 2.0),
        source_frame="MNI152 2009c atlas coordinate mapped once into CT brain-support fractions",
    ),
    transducer=TransducerPoseSpec(
        element_count=1024,
        frequency_hz=650.0e3,
        radius_m=0.150,
        # InsightEC-like hemispherical helmet: elements cover the full calvarium
        # dome including the vertex (cap_min = 0 → elements on top of the head)
        # out to ~80° from the superior axis (cap_max ≈ 1.40 rad → near-complete
        # hemisphere wrapping the calvarium, not a shallow skull-cap).
        cap_min_polar_rad=0.0,
        cap_max_polar_rad=1.40,
        aperture_diameter_m=0.120,
        brain_sound_speed_m_s=1540.0,
        skull_sound_speed_m_s=2800.0,
        target_peak_pressure_pa=1.0e6,
        diagnostic_source_pressure_pa=1.5e5,
        samples_per_ray=192,
        skull_hu_threshold=300.0,
        body_hu_threshold=-350.0,
    ),
)
