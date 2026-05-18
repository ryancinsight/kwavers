"""Domain types for segmentation-driven therapy planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Tissue(IntEnum):
    """Integer labels used by the chapter segmentation."""

    AIR = 0
    NORMAL = 1
    FAT = 2
    BONE = 3
    TUMOR = 4
    AVOID = 5


@dataclass(frozen=True)
class TissueProperties:
    """Acoustic material properties used by the geometric path model."""

    sound_speed_m_s: float
    density_kg_m3: float
    attenuation_np_m: float


TISSUE_PROPERTIES: dict[Tissue, TissueProperties] = {
    Tissue.AIR: TissueProperties(343.0, 1.2, 22.0),
    Tissue.NORMAL: TissueProperties(1540.0, 1060.0, 6.0),
    Tissue.FAT: TissueProperties(1450.0, 920.0, 8.0),
    Tissue.BONE: TissueProperties(2800.0, 1900.0, 35.0),
    Tissue.TUMOR: TissueProperties(1565.0, 1070.0, 7.0),
    Tissue.AVOID: TissueProperties(1540.0, 1060.0, 6.0),
}


@dataclass(frozen=True)
class SegmentationGrid:
    """Label map and Cartesian sampling for a 2-D planning slice."""

    labels: np.ndarray
    body_mask: np.ndarray
    spacing_m: float

    def __post_init__(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError("labels must be a 2-D array")
        if self.body_mask.shape != self.labels.shape:
            raise ValueError("body_mask must match labels")
        if not np.isfinite(self.spacing_m) or self.spacing_m <= 0.0:
            raise ValueError("spacing_m must be finite and positive")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.labels.shape[0]), int(self.labels.shape[1])

    def mask(self, tissue: Tissue) -> np.ndarray:
        return self.labels == int(tissue)

    def coordinates_m(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny = self.shape
        x = (np.arange(nx, dtype=float) - 0.5 * (nx - 1)) * self.spacing_m
        y = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * self.spacing_m
        return np.meshgrid(x, y, indexing="ij")

    def point_from_index(self, index: tuple[int, int]) -> np.ndarray:
        nx, ny = self.shape
        return np.asarray(
            [
                (float(index[0]) - 0.5 * (nx - 1)) * self.spacing_m,
                (float(index[1]) - 0.5 * (ny - 1)) * self.spacing_m,
            ],
            dtype=float,
        )

    def index_from_point(self, point_m: np.ndarray) -> tuple[int, int]:
        nx, ny = self.shape
        i = int(round(float(point_m[0]) / self.spacing_m + 0.5 * (nx - 1)))
        j = int(round(float(point_m[1]) / self.spacing_m + 0.5 * (ny - 1)))
        return i, j

    def sample_points(self, mask: np.ndarray, max_points: int) -> np.ndarray:
        indices = np.argwhere(mask)
        if indices.size == 0 or max_points <= 0:
            return np.zeros((0, 2), dtype=float)
        count = min(int(max_points), int(indices.shape[0]))
        chosen = np.linspace(0, indices.shape[0] - 1, count, dtype=int)
        return np.vstack([self.point_from_index(tuple(idx)) for idx in indices[chosen]])

    def centroid(self, tissue: Tissue) -> np.ndarray:
        points = self.sample_points(self.mask(tissue), max_points=np.count_nonzero(self.mask(tissue)))
        if points.size == 0:
            raise ValueError(f"segmentation has no {tissue.name.lower()} voxels")
        return np.mean(points, axis=0)


@dataclass(frozen=True)
class TransducerAperture:
    """Candidate transducer aperture sampled as discrete elements."""

    angle_deg: float
    center_m: np.ndarray
    inward_unit: np.ndarray
    element_positions_m: np.ndarray
    element_inward_units_m: np.ndarray | None = None
    source_angles_deg: tuple[float, ...] = ()


@dataclass(frozen=True)
class HybridPlanConfig:
    """Weights and solver controls for the chapter hybrid optimizer."""

    frequency_hz: float = 650_000.0
    element_count: int = 10
    aperture_width_m: float = 0.032
    standoff_m: float = 0.016
    candidate_angles_deg: tuple[float, ...] = (
        -170.0,
        -130.0,
        -90.0,
        -50.0,
        -10.0,
        30.0,
        70.0,
        110.0,
        150.0,
    )
    crossfire_aperture_count: int = 3
    crossfire_min_angle_separation_deg: float = 60.0
    max_target_points: int = 36
    max_avoid_points: int = 24
    max_normal_points: int = 48
    max_sidelobe_points: int = 256
    max_hotspot_points: int = 96
    ray_samples: int = 48
    ridge: float = 1.0e-2
    target_weight: float = 1.0
    avoid_weight: float = 6.0
    normal_weight: float = 1.2
    sidelobe_weight: float = 5.0
    hotspot_weight: float = 28.0
    air_path_penalty: float = 4.5
    bone_path_penalty: float = 5.5
    fat_path_penalty: float = 0.7
    protected_peak_penalty: float = 4.0
    off_target_penalty: float = 1.1
    sidelobe_peak_penalty: float = 1.8
    sidelobe_exclusion_radius_m: float = 0.010
    target_dominance_ratio: float = 1.0
    hotspot_min_spacing_m: float = 0.006
    hotspot_refinement_rounds: int = 5
    spot_major_axis_m: float = 0.010
    spot_minor_axis_m: float = 0.0045
    spot_angle_deg: float = -18.0
