"""Array-sensor geometry utilities for k-wave parity examples.

The exact arc construction used by k-wave-python is focus-based:

* `arc_position` is the midpoint of the arc's rear surface.
* `focus_position` is any point on the beam axis.
* `radius` is the arc radius of curvature.
* `diameter` is the chord length.

For a uniform Cartesian grid, translating the whole setup by a fixed vector
preserves the discrete solution because the acoustic equations are translation
invariant on a homogeneous mesh. That allows the centered k-wave coordinate
frame to be mapped into the positive-origin pykwavers frame without changing
the relative source/sensor geometry.

This module keeps the geometry conversion and the array-sensor reduction in one
place so parity examples do not duplicate the same reconstruction logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ArcElementGeometry:
    """Exact pykwavers representation of a focus-based k-wave arc element."""

    kwave_position: tuple[float, float]
    kwave_focus: tuple[float, float]
    pykwavers_center: tuple[float, float, float]
    radius_m: float
    diameter_m: float
    start_angle_deg: float
    end_angle_deg: float
    measure_m: float


def build_arc_element_geometry(
    arc_position_kw: Sequence[float],
    radius_m: float,
    diameter_m: float,
    focus_position_kw: Sequence[float],
    translation_m: Sequence[float],
) -> ArcElementGeometry:
    """Convert k-wave-python arc geometry to the pykwavers arc representation.

    The canonical arc center is

    `c = a + r u`, where `a` is the arc midpoint and
    `u = (f - a) / ||f - a||` is the inward beam unit vector.

    The arc spans the angular interval

    `θ ∈ [arg(-u) - φ, arg(-u) + φ]`

    where `φ = asin(d / (2r))`.
    """

    arc_position_kw = np.asarray(arc_position_kw, dtype=np.float64).reshape(2)
    focus_position_kw = np.asarray(focus_position_kw, dtype=np.float64).reshape(2)
    translation_m = np.asarray(translation_m, dtype=np.float64).reshape(2)

    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if diameter_m <= 0.0:
        raise ValueError("diameter_m must be positive")
    if diameter_m > 2.0 * radius_m:
        raise ValueError("diameter_m must not exceed 2 * radius_m")

    beam_vec = focus_position_kw - arc_position_kw
    beam_norm = float(np.linalg.norm(beam_vec))
    if beam_norm == 0.0:
        raise ValueError("focus_position_kw must differ from arc_position_kw")
    beam_unit = beam_vec / beam_norm

    center_kw = arc_position_kw + radius_m * beam_unit
    center_pk = center_kw + translation_m

    half_angle = float(np.arcsin(diameter_m / (2.0 * radius_m)))
    mid_angle = float(np.arctan2(-beam_unit[1], -beam_unit[0]))

    return ArcElementGeometry(
        kwave_position=(float(arc_position_kw[0]), float(arc_position_kw[1])),
        kwave_focus=(float(focus_position_kw[0]), float(focus_position_kw[1])),
        pykwavers_center=(float(center_pk[0]), float(center_pk[1]), 0.0),
        radius_m=float(radius_m),
        diameter_m=float(diameter_m),
        start_angle_deg=float(np.degrees(mid_angle - half_angle)),
        end_angle_deg=float(np.degrees(mid_angle + half_angle)),
        measure_m=float(2.0 * radius_m * half_angle),
    )


def combine_array_sensor_data(
    sensor_data: np.ndarray,
    sensor_mask: np.ndarray,
    element_weight_masks: Sequence[np.ndarray],
    element_measures_m: Sequence[float],
    *,
    grid_spacing_m: float,
    element_dimension: int = 1,
) -> np.ndarray:
    """Reproduce `kWaveArray.combine_sensor_data` for a precomputed sensor matrix."""

    sensor_matrix = np.asarray(sensor_data, dtype=np.float64)
    if sensor_matrix.ndim != 2:
        raise ValueError(f"sensor_data must be 2-D, got shape {sensor_matrix.shape}")

    sensor_mask_bool = np.asarray(sensor_mask, dtype=bool)
    active_indices = np.flatnonzero(sensor_mask_bool.flatten(order="F"))
    if sensor_matrix.shape[0] != active_indices.size:
        if sensor_matrix.shape[1] == active_indices.size:
            sensor_matrix = sensor_matrix.T
        else:
            raise ValueError(
                f"sensor matrix shape {sensor_matrix.shape} does not match active sensor count {active_indices.size}"
            )

    if len(element_weight_masks) != len(element_measures_m):
        raise ValueError("element_weight_masks and element_measures_m must have the same length")

    combined = np.zeros((len(element_weight_masks), sensor_matrix.shape[1]), dtype=np.float64)
    for element_index, (weights, measure_m) in enumerate(zip(element_weight_masks, element_measures_m)):
        weight_flat = np.asarray(weights, dtype=np.float64).flatten(order="F")
        element_indices = np.flatnonzero(weight_flat)
        if element_indices.size == 0:
            raise ValueError(f"element {element_index} has no active sensor cells")

        local_ind = np.isin(active_indices, element_indices)
        local_weights = weight_flat[element_indices]
        scale = float(measure_m) / (float(grid_spacing_m) ** int(element_dimension))
        combined[element_index] = np.sum(sensor_matrix[local_ind] * local_weights[:, None], axis=0) / scale

    return combined
