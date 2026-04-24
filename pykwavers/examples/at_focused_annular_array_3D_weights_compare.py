#!/usr/bin/env python3
"""
at_focused_annular_array_3D_weights_compare.py
===============================================
Weighted-mask parity check for the ``at_focused_annular_array_3D`` geometry.

Goes one step past the binary-mask compare by validating that pykwavers'
``KWaveArray.get_array_weighted_mask`` for an annular array matches
k-wave-python's ``kWaveArray.get_array_grid_weights`` in *amplitude*, not
just geometric footprint. This is the right parity gate before running a
full forward simulation — if the source weights disagree, the pressure
field will disagree no matter how accurate the solver is.

Outputs:
    output/at_focused_annular_array_3D_weights_compare.png  — side-by-side
        slices through the bowl apex (pykw | kwave | |diff|)
    output/at_focused_annular_array_3D_weights_metrics.txt  — parity metrics

Upstream source of truth:
    external/k-wave-python/examples/at_focused_annular_array_3D/at_focused_annular_array_3D.py

Usage:
    python examples/at_focused_annular_array_3D_weights_compare.py
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.kgrid import kWaveGrid
from kwave.utils.kwave_array import kWaveArray as KWaveArray_Kwave
from kwave.utils.math import round_even

# ---------------------------------------------------------------------------
# Geometry — mirrors at_focused_annular_array_3D.py with a reduced grid so
# the BLI-weighted mask build is fast (~5 s).
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

PNG_PATH = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_weights_compare.png"
REPORT_PATH = DEFAULT_OUTPUT_DIR / "at_focused_annular_array_3D_weights_metrics.txt"

BLI_TOLERANCE = 0.05
UPSAMPLING_RATE = 10


def run_kwave_weights() -> np.ndarray:
    kgrid = kWaveGrid([NX, NY, NZ], [DX, DX, DX])
    karr = KWaveArray_Kwave(bli_tolerance=BLI_TOLERANCE, upsampling_rate=UPSAMPLING_RATE)
    bowl_pos = [float(kgrid.x_vec[0][0]) + SOURCE_X_OFFSET * DX, 0.0, 0.0]
    focus_pos = [float(kgrid.x_vec[-1][0]), 0.0, 0.0]
    karr.add_annular_array(bowl_pos, SOURCE_ROC, DIAMETERS, focus_pos)
    return np.asarray(karr.get_array_grid_weights(kgrid), dtype=float)


def run_pykwavers_weights() -> np.ndarray:
    """Build pykwavers' weighted annular-array mask. See the mask-compare
    sibling example for the `bowl_pos` apex → centre-of-curvature offset
    convention."""
    grid = pkw.Grid(NX, NY, NZ, DX, DX, DX)

    # k-wave's x_vec[i] = (i - (NX-1)/2)*dx, so converting world coords from
    # k-wave (origin-at-domain-centre) to pykwavers (origin-at-corner) uses
    # the half-cell offset (NX-1)/2 * dx. Using NX*dx/2 would misplace the
    # apex by half a cell and smear its BLI weight across two slices.
    offset_x = (NX - 1) / 2.0 * DX
    offset_y = (NY - 1) / 2.0 * DX
    offset_z = (NZ - 1) / 2.0 * DX

    apex_x_kwave = (SOURCE_X_OFFSET - (NX - 1) / 2.0) * DX
    apex_x_world = apex_x_kwave + offset_x
    bowl_pos = (apex_x_world + SOURCE_ROC, offset_y, offset_z)

    diameters = [(float(inner), float(outer)) for inner, outer in DIAMETERS]

    arr = pkw.KWaveArray()
    arr.add_annular_array(bowl_pos, SOURCE_ROC, diameters)
    return np.asarray(arr.get_array_weighted_mask(grid), dtype=float)


def apex_slice_index() -> int:
    """Axial slice index at the bowl apex (where most BLI energy lives)."""
    return SOURCE_X_OFFSET


def save_side_by_side(kw_w: np.ndarray, pkw_w: np.ndarray, ix: int) -> None:
    a = pkw_w[ix, :, :]
    b = kw_w[ix, :, :]
    diff = np.abs(a - b)
    vmax = max(float(a.max()), float(b.max())) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(a.T, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"pykwavers weights (x={ix})\nsum={a.sum():.2f}")
    axes[1].imshow(b.T, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"k-wave weights (x={ix})\nsum={b.sum():.2f}")
    im = axes[2].imshow(diff.T, origin="lower", vmin=0.0, cmap="magma")
    axes[2].set_title(f"|pykw - kwave|\nmax={diff.max():.3g}")
    for ax in axes:
        ax.set_xlabel("y [cell]")
        ax.set_ylabel("z [cell]")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"Annular-array weight parity — slice at x={ix}")
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)


def main() -> int:
    print("[at_focused_annular_array_3D_weights_compare]")
    print(f"  grid: {NX} x {NY} x {NZ} @ dx={DX*1e3:.3f} mm")
    print(f"  rings (inner, outer) [mm]: {[(d[0]*1e3, d[1]*1e3) for d in DIAMETERS]}")

    print("  [k-wave] building weighted mask...")
    kw_w = run_kwave_weights()
    print(f"    weight sum: {kw_w.sum():.4f}  max: {kw_w.max():.4f}")

    print("  [pykwavers] building weighted mask...")
    pkw_w = run_pykwavers_weights()
    print(f"    weight sum: {pkw_w.sum():.4f}  max: {pkw_w.max():.4f}")

    # Full-volume metrics
    vol_m = compute_image_metrics(kw_w, pkw_w)
    # Apex-slice metrics (where the source is concentrated — diff is most visible)
    ix = apex_slice_index()
    slc_m = compute_image_metrics(kw_w[ix], pkw_w[ix])

    print("  --- weighted-mask parity (full volume) ---")
    for k, v in vol_m.items():
        print(f"    {k}: {v}")
    print(f"  --- weighted-mask parity (apex slice x={ix}) ---")
    for k, v in slc_m.items():
        print(f"    {k}: {v}")

    save_side_by_side(kw_w, pkw_w, ix)
    print(f"  image: {PNG_PATH}")

    # Parity targets — physically meaningful gates:
    #   1. Total weight ratio (mass injection). This is the integrated source
    #      power and is the dominant driver of forward-sim amplitude parity.
    #   2. Peak ratio — worst-case cell amplitude.
    #   3. PSNR — noise-floor on cell-level disagreement.
    # Per-cell pearson is a weaker signal here because pykwavers and k-wave
    # choose different integration points on the bowl surface, so BLI stencils
    # land on non-coincident cells even when total injection agrees.
    total_ratio = pkw_w.sum() / (kw_w.sum() + 1e-30)
    total_ok = 0.98 <= total_ratio <= 1.02
    peak_ok = 0.85 <= vol_m["peak_ratio"] <= 1.20
    psnr_ok = vol_m["psnr_db"] >= 25.0
    status = "PASS" if (total_ok and peak_ok and psnr_ok) else "FAIL"
    print(
        f"  total_ratio={total_ratio:.4f}  peak_ratio={vol_m['peak_ratio']:.4f}  "
        f"PSNR={vol_m['psnr_db']:.2f} dB  => {status}"
    )

    lines = [
        f"  grid: {NX} x {NY} x {NZ} dx={DX*1e3:.4f} mm",
        f"  radius of curvature: {SOURCE_ROC*1e3} mm",
        f"  rings (inner, outer) [mm]: {[(d[0]*1e3, d[1]*1e3) for d in DIAMETERS]}",
        "",
        "volume weighted-mask parity",
        "---------------------------",
    ]
    for k, v in vol_m.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"apex-slice (x={ix}) weighted-mask parity")
    lines.append("----------------------------------------")
    for k, v in slc_m.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"status: {status}")
    lines.append(f"image:  {PNG_PATH.name}")
    save_text_report(REPORT_PATH, "at_focused_annular_array_3D_weights_compare", lines)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
