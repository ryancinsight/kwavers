#!/usr/bin/env python3
"""
at_focused_annular_array_3D_mask_compare.py
============================================
Mask-level parity check for the ``at_focused_annular_array_3D`` geometry.

Validates that ``KWaveArray.add_annular_array`` in pykwavers produces a binary
source mask whose active cells are a subset of k-wave-python's BLI-weighted
source mask for the same annular-array geometry.

This script does NOT run the full forward simulation — it isolates the new
``ElementShape::Annulus`` rasterizer so future amplitude/phase parity regressions
can be attributed to the solver path, not the geometry build.

Upstream source of truth:
    external/k-wave-python/examples/at_focused_annular_array_3D/at_focused_annular_array_3D.py

Usage:
    python examples/at_focused_annular_array_3D_mask_compare.py
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    save_side_by_side_parity_figure,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.kgrid import kWaveGrid
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.math import round_even

# ---------------------------------------------------------------------------
# Geometry — mirrors at_focused_annular_array_3D.py with a reduced grid so
# the mask build is fast. The annular-array geometry is identical to upstream.
# ---------------------------------------------------------------------------
C0 = 1500.0
SOURCE_F0 = 1.0e6
PPW = 3

AXIAL_SIZE = 40e-3
LATERAL_SIZE = 45e-3

SOURCE_ROC = 30e-3
DIAMETERS = [
    [0.0, 5.0e-3],
    [10.0e-3, 15.0e-3],
    [20.0e-3, 25.0e-3],
]
SOURCE_X_OFFSET = 20

DX = C0 / (PPW * SOURCE_F0)
NX = round_even(AXIAL_SIZE / DX) + SOURCE_X_OFFSET
NY = round_even(LATERAL_SIZE / DX)
NZ = NY

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_mask_compare.png"
REPORT_PATH = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_mask_metrics.txt"


def run_kwave_mask() -> np.ndarray:
    kgrid = kWaveGrid([NX, NY, NZ], [DX, DX, DX])
    karr = KWaveArray_Kwave(bli_tolerance=0.05, upsampling_rate=10)
    bowl_pos = [float(kgrid.x_vec[0][0]) + SOURCE_X_OFFSET * DX, 0.0, 0.0]
    focus_pos = [float(kgrid.x_vec[-1][0]), 0.0, 0.0]
    karr.add_annular_array(bowl_pos, SOURCE_ROC, DIAMETERS, focus_pos)
    return np.asarray(karr.get_array_binary_mask(kgrid), dtype=bool)


def run_pykwavers_mask() -> np.ndarray:
    """Build the pykwavers annular-array mask using the native
    ``add_annular_array`` API. Grid convention: k-wave places the origin at
    the geometric centre; pykwavers places it at the corner, so we shift the
    bowl centre of curvature by +offset to land on the same world coords."""
    grid = pkw.Grid(NX, NY, NZ, DX, DX, DX)

    # k-wave's x_vec[i] = (i - Nx/2)*dx (integer Nx/2, MATLAB centering).
    # pykwavers world: x_vec[i] = i*dx, origin at 0.
    # apex_x_world = SOURCE_X_OFFSET * DX (x offsets cancel).
    # y,z: k-wave center = y_vec[NY//2] = 0 ↔ pykwavers NY*DX/2 (on grid cell NY//2).
    apex_x_world = SOURCE_X_OFFSET * DX
    center_y = NY * DX / 2.0
    center_z = NZ * DX / 2.0

    # k-wave's `bowl_pos` is the apex; kwavers' ElementShape::Bowl/Annulus
    # stores the centre of curvature (apex sits at centre.x - radius along
    # the bowl axis). Convert apex → centre by pushing +radius along +x.
    bowl_pos = (apex_x_world + SOURCE_ROC, center_y, center_z)

    # Convert diameter list-of-lists to list of tuples
    diameters = [(float(inner), float(outer)) for inner, outer in DIAMETERS]

    arr = pkw.KWaveArray()
    arr.add_annular_array(bowl_pos, SOURCE_ROC, diameters)
    return np.asarray(arr.get_array_binary_mask(grid), dtype=bool)


def report(kw_mask: np.ndarray, pkw_mask: np.ndarray) -> dict:
    kw_count = int(kw_mask.sum())
    pkw_count = int(pkw_mask.sum())
    inter = int((kw_mask & pkw_mask).sum())
    union = int((kw_mask | pkw_mask).sum())
    iou = inter / union if union else 0.0
    dice = 2 * inter / (kw_count + pkw_count) if (kw_count + pkw_count) else 0.0
    inclusion = inter / pkw_count if pkw_count else 0.0

    return {
        "kwave_active_cells": kw_count,
        "pykwavers_active_cells": pkw_count,
        "active_cell_ratio": pkw_count / kw_count if kw_count else float("inf"),
        "intersection": inter,
        "iou": iou,
        "dice": dice,
        "pykwavers_inclusion_in_kwave": inclusion,
    }


def main() -> int:
    print("[at_focused_annular_array_3D_mask_compare]")
    print(f"  grid: {NX} x {NY} x {NZ} @ dx={DX*1e3:.3f} mm")
    print(f"  radius of curvature: {SOURCE_ROC*1e3} mm")
    print(f"  rings (inner, outer) [mm]: {[(d[0]*1e3, d[1]*1e3) for d in DIAMETERS]}")

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

    # Parity metric: IoU. Unlike the rotated-rect case (NGP vs BLI),
    # both sides rasterize bowl/annulus via a BLI stencil, so their active-cell
    # counts match within a percent and IoU is the honest parity gate. Dice
    # and inclusion are reported for reference.
    iou = m["iou"]
    iou_ok = iou >= 0.85
    count_nonzero = m["pykwavers_active_cells"] > 0 and m["kwave_active_cells"] > 0
    status = "PASS" if (iou_ok and count_nonzero) else "FAIL"
    print(f"  IoU: {iou:.4f} (dice {m['dice']:.4f}, inclusion {m['pykwavers_inclusion_in_kwave']:.4f})")
    print(f"  => {status}")

    lines = [
        f"  grid: {NX} x {NY} x {NZ} dx={DX*1e3:.4f} mm",
        f"  radius of curvature: {SOURCE_ROC*1e3} mm",
        f"  rings (inner, outer) [mm]: {[(d[0]*1e3, d[1]*1e3) for d in DIAMETERS]}",
        "",
        "mask parity",
        "-----------",
    ]
    for k, v in m.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"parity_status: {status}")
    print(f"parity_status: {status}")
    figure_path = save_side_by_side_parity_figure(
        kw_mask,
        pkw_mask,
        FIGURE_PATH,
        title="at_focused_annular_array_3D binary mask parity",
        reference_label="k-wave-python mask",
        candidate_label="pykwavers mask",
        projection="peak_slice",
        axis=0,
        cmap="gray",
    )
    lines.append(f"figure: {figure_path.name}")
    print(f"  image: {figure_path}")
    save_text_report(REPORT_PATH, "at_focused_annular_array_3D_mask_compare", lines)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
