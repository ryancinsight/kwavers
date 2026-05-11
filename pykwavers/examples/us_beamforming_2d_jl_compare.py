#!/usr/bin/env python3
"""
us_beamforming_2d_jl_compare.py
===============================
KWave.jl ``beamform_delay_and_sum`` (active-imaging DAS) vs pykwavers
``beamform_image_delay_and_sum`` (active-imaging DAS) for the same sensor
data from a 2-D point-scatterer simulation.

Both engines run the *same* algorithm: per pixel, one-way TOF from the
pixel to each sensor → linear interpolation on the recorded trace →
apodization-weighted coherent sum → divide by the number of contributing
sensors. The kwavers implementation lives at
``kwavers::analysis::signal_processing::beamforming::imaging_das``;
KWave.jl ships the reference at
``KWave.jl/reconstruction/beamform.jl::beamform_delay_and_sum``.

Parity criteria:
    Pearson r (kwavers vs KWave.jl image) >= 0.99
    peak_loc_offset_cells <= 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()
import pykwavers as pkw

REPO_ROOT = HERE.parents[1]
JULIA_PROJECT = REPO_ROOT / "external" / "k-wave-julia" / "KWave.jl"
JULIA_DRIVER = HERE / "run_kwave_julia_us_beamforming_2d.jl"

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH = OUTPUT_DIR / "us_beamforming_2d_jl_compare.png"
METRICS_PATH = OUTPUT_DIR / "us_beamforming_2d_jl_metrics.txt"
JL_SENSOR_CSV = OUTPUT_DIR / "us_beamforming_2d_jl_sensor.csv"
JL_POS_CSV = OUTPUT_DIR / "us_beamforming_2d_jl_positions.csv"
JL_IMAGE_CSV = OUTPUT_DIR / "us_beamforming_2d_jl_image.csv"
JL_META = OUTPUT_DIR / "us_beamforming_2d_jl_meta.json"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
NX, NY = 96, 96
DX = 0.1e-3
C0 = 1500.0
RHO0 = 1000.0
SRC_X_1BASED = NX // 2 + 1
SRC_Y_1BASED = NY // 2 + 1
SENSOR_X_1BASED = 4                  # left-edge linear array
SENSOR_Y_LO_1BASED = 16
SENSOR_Y_HI_1BASED = NY - 16
PML_SIZE = 20

PARITY_THRESHOLDS = {
    "peak_loc_offset_cells": 1,
    "pearson_r_min":         0.99,
}


def run_julia() -> dict:
    julia = os.environ.get("JULIA_BIN", "julia")
    cmd = [
        julia, f"--project={JULIA_PROJECT}", str(JULIA_DRIVER),
        "--nx", str(NX), "--ny", str(NY), "--dx", str(DX),
        "--c0", str(C0), "--rho0", str(RHO0),
        "--src-x-1based", str(SRC_X_1BASED),
        "--src-y-1based", str(SRC_Y_1BASED),
        "--sensor-x-1based", str(SENSOR_X_1BASED),
        "--sensor-y-lo-1based", str(SENSOR_Y_LO_1BASED),
        "--sensor-y-hi-1based", str(SENSOR_Y_HI_1BASED),
        "--pml-size", str(PML_SIZE),
        "--out-sensor-csv", str(JL_SENSOR_CSV),
        "--out-positions-csv", str(JL_POS_CSV),
        "--out-image-csv", str(JL_IMAGE_CSV),
        "--out-meta", str(JL_META),
    ]
    print("[julia] launching:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Julia driver failed (exit {proc.returncode})")

    sensor = np.loadtxt(JL_SENSOR_CSV, delimiter=",")     # (n_sens, nt)
    positions_y = np.loadtxt(JL_POS_CSV)                  # (n_sens,) y in m
    image_jl = np.loadtxt(JL_IMAGE_CSV, delimiter=",").T  # → (Ny_img, Nx_img)
    meta = json.loads(JL_META.read_text())
    return {
        "sensor": sensor,
        "positions_y": positions_y,
        "image": image_jl,
        "meta": meta,
    }


def run_pykwavers_imaging_das(sensor_data: np.ndarray, positions_y: np.ndarray,
                                img_y_m: np.ndarray, img_x_m: np.ndarray,
                                dt: float) -> np.ndarray:
    """Run pykwavers active-imaging DAS on the same sensor data, evaluated at
    the KWave.jl imaging grid pixels. Mirrors KWave.jl beamform_delay_and_sum.
    """
    n_sens = sensor_data.shape[0]
    sensor_positions = np.zeros((n_sens, 3), dtype=np.float64)
    # KWave.jl beamform_delay_and_sum with a 1D sensor_positions vector treats
    # the array as colocated at depth=0 (its sz = zeros). The grid pixel depth
    # img_x_m is therefore the depth-from-array coordinate, not an absolute
    # simulation coordinate. Mirror that convention exactly.
    sensor_positions[:, 0] = 0.0                              # depth (x) — match KWave.jl
    sensor_positions[:, 1] = positions_y                      # lateral (y)
    sensor_positions[:, 2] = 0.0

    grid_pts = []
    for x_m in img_x_m:
        for y_m in img_y_m:
            grid_pts.append([x_m, y_m, 0.0])
    grid_pts = np.array(grid_pts, dtype=np.float64)

    flat = pkw.beamform_image_delay_and_sum(
        np.asarray(sensor_data, dtype=np.float64),
        sensor_positions,
        grid_pts,
        sound_speed=C0,
        sampling_frequency=1.0 / dt,
        apodization="hann",
    )
    flat = np.asarray(flat, dtype=np.float64)
    image = flat.reshape(len(img_x_m), len(img_y_m)).T
    return image


def find_peak_and_contrast(image: np.ndarray) -> tuple[tuple[int, int], float]:
    """Returns ((row, col) of peak, contrast_dB)."""
    abs_img = np.abs(image)
    flat = abs_img.ravel()
    peak_idx = int(np.argmax(flat))
    peak_val = float(flat[peak_idx])
    row, col = np.unravel_index(peak_idx, abs_img.shape)

    # Mask an 11x11 neighbourhood around the peak (covers the main lobe
    # for a typical narrow-aperture DAS) and find the second-largest pixel.
    masked = abs_img.copy()
    rl, rh = max(0, row - 5), min(abs_img.shape[0], row + 6)
    cl, ch = max(0, col - 5), min(abs_img.shape[1], col + 6)
    masked[rl:rh, cl:ch] = 0.0
    second = float(np.max(masked)) + 1e-30
    contrast_db = 20.0 * np.log10(peak_val / second) if peak_val > 0 else 0.0
    return (int(row), int(col)), contrast_db


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    jl = run_julia()
    dt = float(jl["meta"]["dt"])
    img_y_m = np.array(jl["meta"]["img_y_m"], dtype=float)
    img_x_m = np.array(jl["meta"]["img_x_m"], dtype=float)
    print(f"[meta] nt={jl['meta']['nt']} dt={dt:.3e} n_sensors={jl['sensor'].shape[0]}")

    image_jl = jl["image"]   # (Ny_img, Nx_img)
    image_py = run_pykwavers_imaging_das(
        jl["sensor"], jl["positions_y"], img_y_m, img_x_m, dt,
    )

    (rj, cj), contrast_jl = find_peak_and_contrast(image_jl)
    (rp, cp), contrast_py = find_peak_and_contrast(image_py)
    offset_cells = max(abs(rj - rp), abs(cj - cp))
    metrics = compute_image_metrics(image_jl, image_py)
    pearson_r = float(metrics["pearson_r"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    extent = (img_x_m[0] * 1e3, img_x_m[-1] * 1e3,
              img_y_m[0] * 1e3, img_y_m[-1] * 1e3)
    vmax = max(float(np.max(np.abs(image_jl))), float(np.max(np.abs(image_py))))
    axes[0].imshow(image_jl, origin="lower", cmap="seismic",
                    vmin=-vmax, vmax=vmax, extent=extent, aspect="auto")
    axes[0].set_title(
        f"KWave.jl DAS  peak@({rj},{cj})  contrast={contrast_jl:.1f} dB"
    )
    axes[0].set_xlabel("depth x [mm]"); axes[0].set_ylabel("lateral y [mm]")
    axes[1].imshow(image_py, origin="lower", cmap="seismic",
                    vmin=-vmax, vmax=vmax, extent=extent, aspect="auto")
    axes[1].set_title(
        f"kwavers DAS  peak@({rp},{cp})  contrast={contrast_py:.1f} dB"
    )
    axes[1].set_xlabel("depth x [mm]"); axes[1].set_ylabel("lateral y [mm]")
    diff = image_py - image_jl
    dvmax = float(np.max(np.abs(diff))) + 1e-30
    axes[2].imshow(diff, origin="lower", cmap="seismic",
                    vmin=-dvmax, vmax=dvmax, extent=extent, aspect="auto")
    axes[2].set_title(f"kwavers − KWave.jl  Pearson r={pearson_r:.4f}")
    axes[2].set_xlabel("depth x [mm]"); axes[2].set_ylabel("lateral y [mm]")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=140); plt.close(fig)

    pass_fail = (
        offset_cells <= PARITY_THRESHOLDS["peak_loc_offset_cells"]
        and pearson_r >= PARITY_THRESHOLDS["pearson_r_min"]
    )

    lines = [
        f"engine_ref   : KWave.jl/beamform_delay_and_sum (active-imaging DAS)",
        f"engine_cand  : kwavers beamforming::imaging_das (active-imaging DAS)",
        f"forward sim  : KWave.jl/kspace_first_order, point p0 at "
        f"({SRC_X_1BASED},{SRC_Y_1BASED})",
        f"sensor       : linear array at i={SENSOR_X_1BASED}, "
        f"j=[{SENSOR_Y_LO_1BASED},{SENSOR_Y_HI_1BASED}]",
        f"image grid   : {len(img_y_m)}x{len(img_x_m)} px (lateral x depth)",
        f"-- peak localisation --",
        f"  KWave.jl peak idx     : ({rj}, {cj})",
        f"  kwavers  peak idx     : ({rp}, {cp})",
        f"  offset (cells, max)   : {offset_cells}  "
        f"(threshold <= {PARITY_THRESHOLDS['peak_loc_offset_cells']})",
        f"-- pixel-level parity --",
        f"  pearson_r            : {pearson_r:.4f}  "
        f"(threshold >= {PARITY_THRESHOLDS['pearson_r_min']})",
        f"  rms_ratio            : {metrics['rms_ratio']:.4f}",
        f"  peak_ratio           : {metrics['peak_ratio']:.4f}",
        f"  psnr_db              : {metrics['psnr_db']:.2f}",
        f"-- main-lobe contrast (informational) --",
        f"  KWave.jl contrast    : {contrast_jl:.2f} dB",
        f"  kwavers  contrast    : {contrast_py:.2f} dB",
        f"RESULT       : {'PASS' if pass_fail else 'FAIL'}",
    ]
    save_text_report(METRICS_PATH, "us_beamforming_2d_jl_compare", lines)
    print("\n".join(lines))
    print(f"\nFigure : {FIGURE_PATH}")
    print(f"Metrics: {METRICS_PATH}")

    if not pass_fail and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
