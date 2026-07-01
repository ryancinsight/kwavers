#!/usr/bin/env python3
"""
Equivalent-source holography validation for pykwavers ``angular_spectrum_cw``
and ``backward_angular_spectrum_cw``.

Physical setup
--------------
A 2-D Gaussian source with half-width σ = 3 mm and peak amplitude P₀ = 1 Pa
at f₀ = 1 MHz in a lossless medium (c₀ = 1500 m/s).  The Gaussian has all
significant spatial energy within the propagating band (σ_k ≈ 1/σ ≈ 333 rad/m
≪ k = 4189 rad/m), so no evanescent loss occurs.

Two-step computation
--------------------
1. **Forward propagation**: project the source plane to z_m = 20 mm using
   pykwavers ``angular_spectrum_cw`` (spectral propagator H = exp(−j·kz·z)).

2. **Backward propagation (holography)**: reconstruct the source plane from the
   measurement at z_m using ``backward_angular_spectrum_cw`` (conjugate
   propagator H_back = exp(+j·kz·z_m)) with evanescent suppression (Zeng &
   McGough 2008 Eq. 7).  For propagating waves the round-trip product is

       H_back · H_fwd = exp(+j·kz·z_m) · exp(−j·kz·z_m) = 1

   so the reconstruction is exact for all propagating spatial frequencies.

Pass criterion
--------------
Pearson r ≥ 0.9999 and PSNR ≥ 55 dB on |pressure| reconstruction.
The Gaussian source concentrates all significant energy in the propagating band,
so the forward → backward identity holds to floating-point precision.  The
small residual (~0.04% amplitude) is limited by FFT circular-boundary
truncation and zero-padding boundary effects.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    save_text_report,
)

_ROOT = bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
F0 = 1_000_000.0        # source frequency [Hz]
C0 = 1500.0             # sound speed [m/s]
DX = 0.15e-3            # grid spacing [m]  (λ/10 at 1 MHz)
NX = NY = 128           # source plane grid size
SIGMA = 3e-3            # Gaussian half-width [m]; σ_k ≈ 1/σ ≈ 333 rad/m ≪ k
P0 = 1.0                # source pressure amplitude [Pa]
Z_M = 20e-3             # measurement plane distance [m]

PARITY_THRESHOLDS = {
    "pearson_r": 0.9999,
    "psnr_db": 55.0,
}

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "tvsp_equivalent_source_holography_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_equivalent_source_holography_metrics.txt"


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref_max = float(ref.max())
    if ref_max == 0:
        return float("inf")
    mse = float(np.mean((ref - test) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(ref_max**2 / mse)


def save_figure(
    src: np.ndarray,
    p_meas: np.ndarray,
    p_recon: np.ndarray,
    pearson_r: float,
    psnr_db: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    kw = dict(origin="lower", cmap="viridis")

    axes[0].imshow(np.abs(src).T, **kw)
    axes[0].set_title(f"Source |P| (z=0)  [σ={SIGMA*1e3:.0f}mm]")

    axes[1].imshow(np.abs(p_meas).T, **kw)
    axes[1].set_title(f"Forward |P| (z={Z_M*1e3:.0f}mm)")

    im = axes[2].imshow(np.abs(p_recon).T, **kw)
    axes[2].set_title(
        f"Reconstructed |P| (z=0)\n"
        f"Pearson r={pearson_r:.6f}  PSNR={psnr_db:.1f}dB"
    )
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

    fig.suptitle(
        f"Equivalent-Source Holography: Gaussian σ={SIGMA*1e3:.0f}mm"
        f"  f₀={F0/1e6:.1f}MHz  c₀={C0:.0f}m/s  z_m={Z_M*1e3:.0f}mm"
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-failure", action="store_true",
        help="Return exit code 0 even when parity targets are not met.",
    )
    args = parser.parse_args()

    # Build Gaussian source using pykwavers public API
    src = pkw.gaussian_source_2d(NX, NY, DX, SIGMA, amplitude=P0)
    k = 2.0 * np.pi * F0 / C0
    sigma_k = 1.0 / SIGMA  # spatial bandwidth [rad/m]

    print("Computing forward propagation (pykwavers angular_spectrum_cw)...")
    pressure_3d = pkw.angular_spectrum_cw(
        input_plane=src.real,
        dx=DX,
        z_pos=[Z_M],
        f0=F0,
        medium=C0,
        angular_restriction=True,
    )
    p_meas = pressure_3d[:, :, 0]  # (NX, NY)

    print("Computing backward propagation (pykwavers backward_angular_spectrum_cw)...")
    p_recon = pkw.backward_angular_spectrum_cw(
        measurement_plane=p_meas,
        dx=DX,
        z_m=Z_M,
        f0=F0,
        medium=C0,
        angular_restriction=True,
    )

    src_abs   = np.abs(src)
    recon_abs = np.abs(p_recon)

    pearson_r, _ = pearsonr(src_abs.ravel(), recon_abs.ravel())
    psnr_db = _psnr(src_abs, recon_abs)
    rms_ratio = (
        float(np.sqrt(np.mean(recon_abs**2)) / np.sqrt(np.mean(src_abs**2)))
        if src_abs.any() else 1.0
    )
    max_abs_err = float(np.max(np.abs(recon_abs - src_abs)))

    status = (
        "PASS"
        if pearson_r >= PARITY_THRESHOLDS["pearson_r"]
        and psnr_db >= PARITY_THRESHOLDS["psnr_db"]
        else "FAIL"
    )

    report_lines = [
        "tvsp_equivalent_source_holography parity report",
        "================================================",
        "",
        f"Source: 2-D Gaussian, σ={SIGMA*1e3:.1f} mm",
        f"Frequency: f₀={F0/1e6:.1f} MHz,  k={k:.1f} rad/m",
        f"Spatial bandwidth: σ_k ≈ 1/σ = {sigma_k:.0f} rad/m  (≪ k = {k:.0f} rad/m)",
        f"Medium: c₀={C0:.0f} m/s (lossless)",
        f"Grid: {NX}×{NY},  dx={DX*1e3:.3f} mm",
        f"Measurement plane: z_m = {Z_M*1e3:.0f} mm",
        "",
        "Method: backward angular spectrum (conjugate propagator, pkw.backward_angular_spectrum_cw)",
        "  H_fwd(kx,ky) = exp(−j·kz·z_m),   H_back = exp(+j·kz·z_m)",
        "  H_back · H_fwd = 1 for propagating waves → exact round-trip identity",
        "",
        "|Pressure| reconstruction parity metrics",
        "----------------------------------------",
        f"  pearson_r:   {pearson_r:.6f}  (threshold ≥ {PARITY_THRESHOLDS['pearson_r']})",
        f"  psnr_db:     {psnr_db:.2f} dB  (threshold ≥ {PARITY_THRESHOLDS['psnr_db']} dB)",
        f"  rms_ratio:   {rms_ratio:.6f}",
        f"  max_abs_err: {max_abs_err:.4e} Pa",
        "",
        f"parity_status: {status}",
    ]

    for line in report_lines:
        print(line)

    save_figure(src, p_meas, p_recon, pearson_r, psnr_db, FIGURE_PATH)
    save_text_report(
        METRICS_PATH, "tvsp_equivalent_source_holography parity report", report_lines
    )
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {METRICS_PATH}")

    if status == "FAIL" and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
