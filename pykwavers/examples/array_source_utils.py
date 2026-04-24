"""Source-array utilities for k-wave parity examples.

This module keeps the active-point row ordering and source-distribution logic in
one place so the example scripts do not duplicate the same geometry-to-signal
mapping. Pressure-source rows follow MATLAB / Fortran-order active-point
enumeration so the example matrices match k-wave-python.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import pykwavers as pkw

from array_sensor_utils import ArcElementGeometry


def _arc_element_array(geom: ArcElementGeometry) -> pkw.KWaveArray:
    array = pkw.KWaveArray()
    array.add_arc_element(
        geom.pykwavers_center,
        geom.radius_m,
        geom.diameter_m,
        geom.start_angle_deg,
        geom.end_angle_deg,
    )
    return array


def reorder_active_rows(mask: np.ndarray, matrix: np.ndarray, *, from_order: str, to_order: str) -> np.ndarray:
    """Reorder active-point rows by the flattened coordinate order of a mask."""
    if from_order not in {"C", "F"} or to_order not in {"C", "F"}:
        raise ValueError("from_order and to_order must be 'C' or 'F'")

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim == 3 and mask_bool.shape[2] == 1:
        mask_bool = np.squeeze(mask_bool, axis=2)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected a 2-D active mask, got shape {mask_bool.shape}")

    arr = np.asarray(matrix, dtype=np.float64)
    active_coords = np.argwhere(mask_bool)
    n_active = active_coords.shape[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {arr.shape}")
    if arr.shape[0] != n_active:
        if arr.shape[1] == n_active:
            arr = arr.T
        else:
            raise ValueError(f"Matrix row count {arr.shape[0]} does not match active mask count {n_active}")

    if from_order == to_order:
        return arr

    from_linear = np.ravel_multi_index(active_coords.T, mask_bool.shape, order=from_order)
    to_linear = np.ravel_multi_index(active_coords.T, mask_bool.shape, order=to_order)
    from_coords = active_coords[np.argsort(from_linear, kind="stable")]
    to_coords = active_coords[np.argsort(to_linear, kind="stable")]
    from_index = {tuple(coord): idx for idx, coord in enumerate(from_coords)}
    perm = np.array([from_index[tuple(coord)] for coord in to_coords], dtype=np.int64)
    return arr[perm]


def build_pykwavers_distributed_arc_signal(
    arc_geometries: Sequence[ArcElementGeometry],
    grid: pkw.Grid,
    source_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build the binary mask and row-wise active-point signal matrix for pykwavers.

    The returned source signal rows follow Fortran-order over the active mask so
    they match k-wave-python's `matlab_find` convention and the pressure-source
    iteration order used by pykwavers.
    """
    source_signal_arr = np.asarray(source_signal, dtype=np.float64)
    if source_signal_arr.ndim != 2:
        raise ValueError(f"Expected a 2-D element signal matrix, got shape {source_signal_arr.shape}")
    if source_signal_arr.shape[0] != len(arc_geometries):
        raise ValueError(
            "source_signal row count must match the number of arc geometries"
        )

    array = pkw.KWaveArray()
    for geom in arc_geometries:
        array.add_arc_element(
            geom.pykwavers_center,
            geom.radius_m,
            geom.diameter_m,
            geom.start_angle_deg,
            geom.end_angle_deg,
        )

    binary_mask = np.squeeze(np.asarray(array.get_array_binary_mask(grid), dtype=bool))
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2-D binary mask, got shape {binary_mask.shape}")

    active_indices = np.flatnonzero(binary_mask.flatten(order="F"))
    distributed = np.zeros((active_indices.size, source_signal_arr.shape[1]), dtype=np.float64)
    element_weight_masks: list[np.ndarray] = []

    for idx, geom in enumerate(arc_geometries):
        element_array = _arc_element_array(geom)
        weight_mask = np.squeeze(np.asarray(element_array.get_array_weighted_mask(grid), dtype=np.float64))
        if weight_mask.shape != binary_mask.shape:
            raise ValueError(
                f"Element weight mask shape {weight_mask.shape} does not match source mask shape {binary_mask.shape}"
            )
        element_weight_masks.append(weight_mask)

        weight_flat = weight_mask.flatten(order="F")
        element_indices = np.flatnonzero(weight_flat)
        local_ind = np.isin(active_indices, element_indices)
        distributed[local_ind] += weight_flat[element_indices][:, None] * source_signal_arr[idx, :][None, :]

    return binary_mask, distributed, element_weight_masks
