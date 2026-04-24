#!/usr/bin/env python3
"""
at_linear_array_transducer_mask_compare.py
==========================================
Mask-level parity check for the `at_linear_array_transducer` geometry.

Validates that `KWaveArray.add_rect_rot_element` in pykwavers produces a
binary source mask that matches k-wave-python's `kWaveArray.add_rect_element`
with per-element Euler rotation and a shared global `set_array_position`
transform.

This script does NOT run the full forward simulation — it isolates the new
rotated-rect rasterizer so any future amplitude/phase parity regressions can
be attributed to the solver path, not the geometry build.

Upstream source of truth:
    external/k-wave-python/examples/at_linear_array_transducer/at_linear_array_transducer.py

Usage:
    python examples/at_linear_array_transducer_mask_compare.py
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
import kwave.data as kdata
from kwave.kgrid import kWaveGrid
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave

# ---------------------------------------------------------------------------
# Geometry — mirrors at_linear_array_transducer.py with a reduced grid so the
# mask build is fast. The geometry logic (per-element euler + global
# transform composition) is identical.
# ---------------------------------------------------------------------------
C0 = 1500.0
SOURCE_F0 = 1.0e6
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

REPORT_PATH = DEFAULT_OUTPUT_DIR / "at_linear_array_transducer_mask_metrics.txt"


def rotation_matrix_deg(euler_xyz_deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(euler_xyz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    My = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Mz @ My @ Mx


def compose_euler_xyz_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Return intrinsic X-Y-Z Euler angles (degrees) equivalent to R(a)*R(b)."""
    R = rotation_matrix_deg(a_deg) @ rotation_matrix_deg(b_deg)
    # Recover intrinsic X-Y-Z Euler: R = Rz*Ry*Rx
    # ry = asin(R[0,2]); rx = atan2(-R[1,2], R[2,2]); rz = atan2(-R[0,1], R[0,0])
    ry = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    if np.abs(np.cos(ry)) > 1e-9:
        rx = np.arctan2(-R[1, 2], R[2, 2])
        rz = np.arctan2(-R[0, 1], R[0, 0])
    else:
        rx = np.arctan2(R[2, 1], R[1, 1])
        rz = 0.0
    return np.rad2deg(np.array([rx, ry, rz]))


def element_x_positions() -> np.ndarray:
    return (
        -(ELEMENT_NUM * ELEMENT_PITCH / 2 - ELEMENT_PITCH / 2)
        + np.arange(ELEMENT_NUM) * ELEMENT_PITCH
    )


def run_kwave_mask() -> np.ndarray:
    kgrid = kWaveGrid([NX, NY, NZ], [DX, DX, DX])
    karr = KWaveArray_Kwave(bli_tolerance=0.05, upsampling_rate=10)
    z0 = float(kgrid.z_vec[0][0])
    for x_pos in element_x_positions():
        karr.add_rect_element(
            [float(x_pos), 0.0, z0],
            ELEMENT_WIDTH,
            ELEMENT_LENGTH,
            kdata.Vector(ROTATION_DEG.tolist()),
        )
    karr.set_array_position(kdata.Vector(TRANSLATION.tolist()), kdata.Vector(ROTATION_DEG.tolist()))
    return np.asarray(karr.get_array_binary_mask(kgrid), dtype=bool)


def run_pykwavers_mask() -> np.ndarray:
    """Build the pykwavers rotated-rect mask using the native
    ``set_array_position`` global transform. Elements are stored in the
    k-wave-centred local frame; the transform bakes in the translation plus
    the pykwavers origin-at-corner grid offset."""
    grid = pkw.Grid(NX, NY, NZ, DX, DX, DX)

    # Convert k-wave centered coordinates to pykwavers' corner-origin grid.
    offset = np.array(
        [NX * DX / 2.0, NY * DX / 2.0, NZ * DX / 2.0],
        dtype=np.float64,
    )
    z_local_kwave = -NZ * DX / 2.0

    arr = pkw.KWaveArray()
    for x_pos in element_x_positions():
        arr.add_rect_rot_element(
            (float(x_pos), 0.0, z_local_kwave),
            (ELEMENT_WIDTH, ELEMENT_LENGTH, DX),  # single-cell thickness
            tuple(ROTATION_DEG.tolist()),
        )
    arr.set_array_position(
        tuple((TRANSLATION + offset).tolist()),
        tuple(ROTATION_DEG.tolist()),
    )
    return np.asarray(arr.get_array_binary_mask(grid), dtype=bool)


def report(kw_mask: np.ndarray, pkw_mask: np.ndarray) -> dict:
    kw_count = int(kw_mask.sum())
    pkw_count = int(pkw_mask.sum())
    inter = int((kw_mask & pkw_mask).sum())
    union = int((kw_mask | pkw_mask).sum())
    iou = inter / union if union else 0.0
    dice = 2 * inter / (kw_count + pkw_count) if (kw_count + pkw_count) else 0.0

    metrics = {
        "kwave_active_cells": kw_count,
        "pykwavers_active_cells": pkw_count,
        "active_cell_ratio": pkw_count / kw_count if kw_count else float("inf"),
        "intersection": inter,
        "iou": iou,
        "dice": dice,
    }
    return metrics


def main() -> int:
    print("[at_linear_array_transducer_mask_compare]")
    print(f"  grid: {NX} x {NY} x {NZ} @ dx={DX*1e3:.3f} mm")
    print(f"  elements: {ELEMENT_NUM}, pitch={ELEMENT_PITCH*1e3} mm, tilt={ROTATION_DEG} deg")
    print(f"  translation: {TRANSLATION*1e3} mm")

    print("  [k-wave] building mask...")
    kw_mask = run_kwave_mask()
    print(f"    active cells: {int(kw_mask.sum())}")

    print("  [pykwavers] building mask...")
    pkw_mask = run_pykwavers_mask()
    print(f"    active cells: {int(pkw_mask.sum())}")

    m = report(kw_mask, pkw_mask)
    print("  --- mask parity ---")
    for k, v in m.items():
        print(f"    {k}: {v}")

    # Inclusion: every pykwavers cell should land inside k-wave's (wider) BLI
    # footprint. k-wave's `get_array_binary_mask` spreads each off-grid sample
    # over a BLI stencil, so its count is systematically larger than the NGP
    # pykwavers mask. Parity here means geometric placement matches — amplitude
    # parity is handled downstream by the weighted-mask / distributed-signal
    # path, not the binary mask.
    inclusion = m["intersection"] / m["pykwavers_active_cells"] if m["pykwavers_active_cells"] else 0.0
    m["pykwavers_inclusion_in_kwave"] = inclusion
    inclusion_ok = inclusion >= 0.98
    count_nonzero = m["pykwavers_active_cells"] > 0 and m["kwave_active_cells"] > 0
    status = "PASS" if (inclusion_ok and count_nonzero) else "FAIL"
    print(f"  inclusion (pykw ⊆ kwave): {inclusion:.4f}")
    print(f"  => {status}")

    lines = [
        f"  grid: {NX} x {NY} x {NZ} dx={DX*1e3:.4f} mm",
        f"  elements: {ELEMENT_NUM}, pitch={ELEMENT_PITCH*1e3} mm",
        f"  per-element euler: {ROTATION_DEG.tolist()} deg",
        f"  global translation: {TRANSLATION.tolist()} m",
        f"  global rotation:    {ROTATION_DEG.tolist()} deg",
        "",
        "mask parity",
        "-----------",
    ]
    for k, v in m.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"status: {status}")
    save_text_report(REPORT_PATH, "at_linear_array_transducer_mask_compare", lines)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
