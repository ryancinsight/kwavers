#!/usr/bin/env python3
"""Shared geometry and drive helpers for the linear-array transducer example."""

from __future__ import annotations

import numpy as np

import pykwavers as pkw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.signals import tone_burst


C0 = 1500.0
SOURCE_F0 = 1.0e6
SOURCE_CYCLES = 5
SOURCE_AMP = 1.0e6
PPW = 3

GRID_SIZE_X = 40e-3
GRID_SIZE_Y = 20e-3
GRID_SIZE_Z = 40e-3

ELEMENT_NUM = 15
ELEMENT_WIDTH = 1e-3
ELEMENT_LENGTH = 10e-3
ELEMENT_PITCH = 2e-3

TRANSLATION = np.array([5e-3, 0.0, 8e-3], dtype=np.float64)
ROTATION_DEG = np.array([0.0, 20.0, 0.0], dtype=np.float64)

DX = C0 / (PPW * SOURCE_F0)
NX = int(round(GRID_SIZE_X / DX))
NY = int(round(GRID_SIZE_Y / DX))
NZ = int(round(GRID_SIZE_Z / DX))


def rotation_matrix_deg(euler_xyz_deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(euler_xyz_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mz @ my @ mx


def compose_euler_xyz_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Return intrinsic X-Y-Z Euler angles equivalent to `R(a) * R(b)`."""
    r = rotation_matrix_deg(a_deg) @ rotation_matrix_deg(b_deg)
    ry = np.arcsin(np.clip(r[0, 2], -1.0, 1.0))
    if np.abs(np.cos(ry)) > 1.0e-9:
        rx = np.arctan2(-r[1, 2], r[2, 2])
        rz = np.arctan2(-r[0, 1], r[0, 0])
    else:
        rx = np.arctan2(r[2, 1], r[1, 1])
        rz = 0.0
    return np.rad2deg(np.array([rx, ry, rz], dtype=np.float64))


def element_x_positions() -> np.ndarray:
    """Return the centered x-positions used by the 15-element array."""
    return -(ELEMENT_NUM * ELEMENT_PITCH / 2.0 - ELEMENT_PITCH / 2.0) + np.arange(ELEMENT_NUM) * ELEMENT_PITCH


def build_kwave_array(kgrid: kWaveGrid) -> KWaveArray_Kwave:
    """Build the reference k-wave-python array."""
    karr = KWaveArray_Kwave(bli_tolerance=0.05, upsampling_rate=10)
    z0 = float(kgrid.z_vec[0][0])
    for x_pos in element_x_positions():
        karr.add_rect_element(
            [float(x_pos), 0.0, z0],
            ELEMENT_WIDTH,
            ELEMENT_LENGTH,
            ROTATION_DEG.tolist(),
        )
    karr.set_array_position(TRANSLATION.tolist(), ROTATION_DEG.tolist())
    return karr


def build_pykwavers_array() -> pkw.KWaveArray:
    """Build the native pykwavers array in the corner-origin frame."""
    arr = pkw.KWaveArray()
    z_local_kwave = -NZ * DX / 2.0
    for x_pos in element_x_positions():
        arr.add_rect_rot_element(
            (float(x_pos), 0.0, z_local_kwave),
            (ELEMENT_WIDTH, ELEMENT_LENGTH, DX),
            tuple(ROTATION_DEG.tolist()),
        )
    offset = np.array(
        [
            NX * DX / 2.0,
            NY * DX / 2.0,
            NZ * DX / 2.0,
        ],
        dtype=np.float64,
    )
    arr.set_array_position(tuple((TRANSLATION + offset).tolist()), tuple(ROTATION_DEG.tolist()))
    return arr


def build_source_signal_matrix(kgrid: kWaveGrid) -> np.ndarray:
    """Return the per-element delayed tone-burst matrix used by the example.

    The returned matrix has shape ``(ELEMENT_NUM, kgrid.Nt)`` so the source
    contract matches both k-wave-python and pykwavers, which both require
    source signals to span the solver time grid.
    """
    nt = int(kgrid.Nt)
    signals = np.zeros((ELEMENT_NUM, nt), dtype=np.float64)
    if ELEMENT_NUM % 2 != 0:
        centering_offset = np.ceil(ELEMENT_NUM / 2)
    else:
        centering_offset = (ELEMENT_NUM + 1) / 2

    positional_basis = np.arange(1, ELEMENT_NUM + 1) - centering_offset
    source_focus = 20e-3
    time_delays = -(np.sqrt((positional_basis * ELEMENT_PITCH) ** 2 + source_focus**2) - source_focus) / C0
    time_delays = time_delays - np.min(time_delays)

    burst = SOURCE_AMP * tone_burst(
        1.0 / float(kgrid.dt),
        SOURCE_F0,
        SOURCE_CYCLES,
        signal_offset=np.round(time_delays / float(kgrid.dt)).astype(int),
    )
    burst = np.asarray(burst, dtype=np.float64)
    if burst.ndim == 1:
        burst = burst.reshape(1, -1)
    n_cols = min(nt, burst.shape[1])
    signals[:, :n_cols] = burst[:, :n_cols]
    return signals


def build_pykwavers_source_matrix_and_masks(
    grid: pkw.Grid,
    element_signals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return the binary/weighted masks and per-point signal matrix for pykwavers.

    The returned source rows follow Fortran-order active-cell enumeration so the
    matrix matches k-wave-python's pressure-source contract.
    """
    py_array = build_pykwavers_array()
    source_binary_mask = np.asarray(py_array.get_array_binary_mask(grid), dtype=bool)
    source_weighted_mask = np.asarray(py_array.get_array_weighted_mask(grid), dtype=np.float64)
    element_signals = np.asarray(element_signals, dtype=np.float64)

    element_weight_masks: list[np.ndarray] = []
    for x_pos in element_x_positions():
        elem = pkw.KWaveArray()
        elem.add_rect_rot_element(
            (float(x_pos), 0.0, -NZ * DX / 2.0),
            (ELEMENT_WIDTH, ELEMENT_LENGTH, DX),
            tuple(ROTATION_DEG.tolist()),
        )
        elem.set_array_position(
            tuple((TRANSLATION + np.array([NX * DX / 2.0, NY * DX / 2.0, NZ * DX / 2.0])).tolist()),
            tuple(ROTATION_DEG.tolist()),
        )
        element_weight_masks.append(np.asarray(elem.get_array_weighted_mask(grid), dtype=np.float64))

    flat_mask = source_binary_mask.flatten(order="F")
    active_flat = np.flatnonzero(flat_mask)
    active_coords = np.argwhere(flat_mask)
    signal_matrix = np.zeros((active_flat.size, element_signals.shape[1]), dtype=np.float64)

    for row, flat_idx in enumerate(active_flat):
        coord = np.unravel_index(flat_idx, source_binary_mask.shape, order="F")
        for elem_idx, weight_mask in enumerate(element_weight_masks):
            weight = float(weight_mask[coord])
            if weight != 0.0:
                signal_matrix[row] += weight * element_signals[elem_idx]

    return source_binary_mask, source_weighted_mask, signal_matrix, active_coords, element_weight_masks


def build_sensor_mask(nx: int, ny: int, nz: int) -> np.ndarray:
    """Return the on-axis sensor plane used by the linear-array example."""
    sensor_mask = np.zeros((nx, ny, nz), dtype=bool)
    sensor_mask[:, ny // 2, :] = True
    return sensor_mask
