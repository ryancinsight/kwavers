#!/usr/bin/env python3
"""
pr_2D_TR_directional_sensors_compare.py
========================================
Validates 2D directional sensor weighting for CW acoustic field measurement.

Physical setup
--------------
A 2-D Gaussian CW source (σ=3 mm, f₀=1 MHz) forward-propagated to a sensor
plane at z_m=20 mm, then mixed with pixel-uniform in-band noise (krad ≤ k in
the NX×NY k-space, 1% RMS of signal RMS).

Directional weighting theorem
-----------------------------
The cardioid directional filter for a sensor facing +z:

    P_dir(kx,ky) = P(kx,ky) × (1 + kz/k) / 2,   kz = sqrt(k² − kx² − ky²)

Weight profile vs incident angle θ (from z-axis):
    θ=0°: weight=1.0  (on-axis, full signal)
    θ=60°: weight=0.75
    θ=90°: weight=0.5 (grazing, 6 dB reduction)

A Gaussian source with σ_k ≪ k has virtually all energy near kz ≈ k
(weight ≈ 1), so the signal passes intact.

Analytic noise suppression for pixel-uniform in-band forward-propagating noise
-------------------------------------------------------------------------------
With noise amplitude uniform per k-space pixel over the in-band disk
krad ∈ [0, k] (all forward-propagating, kz > 0):

    E[weight²] = 2∫₀^1 [(1+√(1−u²))/2]² u du = 17/24
    → noise power reduced by −10·log10(17/24) ≈ 1.50 dB analytically.

Note: the larger 4.26 dB gain (from the (1/π)∫₀^π formula) requires a true
cardioid sensor combining pressure + particle-velocity to separate forward from
backward components; a k-space filter on pressure alone cannot achieve this for
forward-only in-band noise.

Parity criterion (measurement plane PSNR comparison)
-----------------------------------------------------
  PSNR(P_clean, P_dir)  ≥ PSNR(P_clean, P_noisy) + 1 dB
  PSNR(P_clean, P_dir)  ≥ 45 dB

Outputs
-------
  output/pr_2D_TR_directional_sensors_compare.png
  output/pr_2D_TR_directional_sensors_metrics.txt
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

sys.path.insert(0, str(Path(__file__).parent))
from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    save_text_report,
)

bootstrap_example_paths()
import pykwavers as pkw

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
F0    = 1_000_000.0   # [Hz]
C0    = 1500.0        # [m/s]
DX    = 0.15e-3       # [m]
NX    = NY = 128
SIGMA = 3e-3          # [m]  σ_k = 333 rad/m ≪ k = 4189 rad/m
P0    = 1.0           # [Pa]
Z_M   = 20e-3         # [m]
NOISE_FRACTION = 0.01  # noise RMS = 1% of signal RMS (PSNR_omni ≈ 50 dB)
RNG_SEED = 42

TOL_PSNR_DIR_DB  = 45.0  # [dB] minimum PSNR of directional measurement
TOL_PSNR_GAIN_DB = 1.0   # [dB] minimum gain; analytic ≈ 1.50 dB for forward-only in-band noise

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "pr_2D_TR_directional_sensors_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "pr_2D_TR_directional_sensors_metrics.txt"


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref_max = float(np.abs(ref).max())
    if ref_max == 0:
        return float("inf")
    mse = float(np.mean(np.abs(ref - test) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(ref_max ** 2 / mse)


def _k_grid(nx: int, ny: int, dx: float, k: float):
    """Wavenumber grids in the NX×NY k-space (no padding, FFT-order)."""
    kx_vec = np.fft.fftfreq(nx, d=dx / (2.0 * np.pi))
    ky_vec = np.fft.fftfreq(ny, d=dx / (2.0 * np.pi))
    ky, kx = np.meshgrid(ky_vec, kx_vec, indexing="ij")
    kz = np.sqrt((k**2 - kx**2 - ky**2).astype(complex))
    return kx, ky, kz


def _directional_filter_2d(
    p: np.ndarray, dx: float, f0: float, c0: float
) -> np.ndarray:
    """Apply (1 + kz/k) / 2 directional weighting in the NX×NY k-space."""
    k = 2.0 * np.pi * f0 / c0
    _, _, kz = _k_grid(*p.shape, dx, k)
    P_fft = np.fft.fft2(p)
    weight = (1.0 + kz / k) / 2.0
    return np.fft.ifft2(P_fft * weight)


def _isotropic_inband_noise(
    nx: int, ny: int, dx: float, k: float, rms_target: float, rng
) -> np.ndarray:
    """Generate complex isotropic in-band noise in the NX×NY k-space."""
    kx_vec = np.fft.fftfreq(nx, d=dx / (2.0 * np.pi))
    ky_vec = np.fft.fftfreq(ny, d=dx / (2.0 * np.pi))
    ky, kx = np.meshgrid(ky_vec, kx_vec, indexing="ij")
    in_band = np.sqrt(kx**2 + ky**2) <= k

    # White complex noise spectrum
    amp = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    amp[~in_band] = 0.0
    noise = np.fft.ifft2(amp)
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2))) + 1e-20
    return noise * (rms_target / noise_rms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    k = 2.0 * np.pi * F0 / C0
    # Analytic noise power gain for pixel-uniform in-band forward-propagating noise:
    # 2∫₀^1 [(1+√(1−u²))/2]² u du = 17/24  →  gain = -10·log10(17/24) ≈ 1.50 dB
    expected_gain_db = -10.0 * np.log10(17.0 / 24.0)
    print(
        f"pr_2D_TR_directional_sensors:\n"
        f"  Grid {NX}×{NY}, dx={DX*1e3:.2f} mm, f₀={F0/1e6:.1f} MHz, k={k:.0f} rad/m\n"
        f"  Source σ={SIGMA*1e3:.1f} mm (σ_k={1/SIGMA:.0f} rad/m ≪ k)\n"
        f"  z_m={Z_M*1e3:.0f} mm, noise_fraction={NOISE_FRACTION:.0%} (isotropic in-band)\n"
        f"  Analytically expected noise power gain: {expected_gain_db:.2f} dB"
    )

    # --- Source and forward propagation ---
    src = pkw.gaussian_source_2d(NX, NY, DX, SIGMA, amplitude=P0)
    p_clean = pkw.angular_spectrum_cw(
        src.real, DX, [Z_M], F0, C0, angular_restriction=True,
    )[:, :, 0]

    # --- Isotropic in-band noise (NX×NY k-space) ---
    rng = np.random.default_rng(RNG_SEED)
    signal_rms = float(np.sqrt(np.mean(np.abs(p_clean) ** 2)))
    noise_rms_target = NOISE_FRACTION * signal_rms
    noise = _isotropic_inband_noise(NX, NY, DX, k, noise_rms_target, rng)
    p_noisy = p_clean + noise

    # --- Directional filter (same NX×NY k-space as noise) ---
    p_dir = _directional_filter_2d(p_noisy, DX, F0, C0)

    # --- Metrics ---
    psnr_omni = _psnr(p_clean, p_noisy)
    psnr_dir  = _psnr(p_clean, p_dir)
    psnr_gain = psnr_dir - psnr_omni

    passed = psnr_dir >= TOL_PSNR_DIR_DB and psnr_gain >= TOL_PSNR_GAIN_DB
    status = "PASS" if passed else "FAIL"

    print(f"\n  2D directional sensor comparison [{status}]:")
    print(f"    PSNR (omni noisy vs clean):    {psnr_omni:.2f} dB")
    print(f"    PSNR (directional vs clean):   {psnr_dir:.2f} dB  (threshold ≥ {TOL_PSNR_DIR_DB})")
    print(f"    PSNR gain:                     {psnr_gain:+.2f} dB  (threshold ≥ {TOL_PSNR_GAIN_DB})")
    print(f"    Expected analytic gain:        {expected_gain_db:.2f} dB")

    # Backward reconstruction for visualization
    p_recon_clean = pkw.backward_angular_spectrum_cw(
        p_clean, DX, Z_M, F0, C0, angular_restriction=True,
    )
    p_recon_omni = pkw.backward_angular_spectrum_cw(
        p_noisy, DX, Z_M, F0, C0, angular_restriction=True,
    )
    p_recon_dir = pkw.backward_angular_spectrum_cw(
        p_dir, DX, Z_M, F0, C0, angular_restriction=True,
    )

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    kw = dict(origin="lower", cmap="viridis")

    axes[0, 0].imshow(np.abs(src).T, **kw)
    axes[0, 0].set_title(f"Source (z=0, σ={SIGMA*1e3:.0f}mm)")

    axes[0, 1].imshow(np.abs(p_noisy).T, **kw)
    axes[0, 1].set_title(f"Omnidirectional sensor\n(+{NOISE_FRACTION:.0%} RMS noise)")

    axes[0, 2].imshow(np.abs(p_dir).T, **kw)
    axes[0, 2].set_title(f"Directional sensor\n(1+kz/k)/2 filter")

    axes[1, 0].imshow(np.abs(p_recon_clean).T, **kw)
    axes[1, 0].set_title("Backward recon (clean)")

    axes[1, 1].imshow(np.abs(p_recon_omni).T, **kw)
    axes[1, 1].set_title(f"Backward recon (omni) PSNR={psnr_omni:.1f}dB")

    axes[1, 2].imshow(np.abs(p_recon_dir).T, **kw)
    axes[1, 2].set_title(f"Backward recon (directional) [{status}]\nPSNR={psnr_dir:.1f}dB gain={psnr_gain:+.1f}dB")

    for row in axes:
        for ax in row:
            ax.set_xlabel("x index"); ax.set_ylabel("y index")

    fig.suptitle(
        f"2D Directional Sensor Weighting [{status}]\n"
        f"f₀={F0/1e6:.1f}MHz  σ={SIGMA*1e3:.0f}mm  z_m={Z_M*1e3:.0f}mm  noise={NOISE_FRACTION:.0%}RMS\n"
        f"PSNR gain={psnr_gain:+.2f}dB (≥{TOL_PSNR_GAIN_DB}dB, analytic≈{expected_gain_db:.1f}dB)  "
        f"PSNR_dir={psnr_dir:.1f}dB (≥{TOL_PSNR_DIR_DB}dB)"
    )
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")
    print(f"parity_status: {status}")

    save_text_report(METRICS_PATH, "pr_2D_TR_directional_sensors_compare", [
        f"Status:                     {status}",
        f"psnr_omni_db:               {psnr_omni:.4f}",
        f"psnr_dir_db:                {psnr_dir:.4f}  (threshold ≥ {TOL_PSNR_DIR_DB})",
        f"psnr_gain_db:               {psnr_gain:.4f}  (threshold ≥ {TOL_PSNR_GAIN_DB})",
        f"expected_gain_analytic_db:  {expected_gain_db:.4f}",
        f"noise_fraction:             {NOISE_FRACTION}",
        f"noise_type:                 isotropic_in_band (krad<=k, NXxNY k-space)",
        f"sigma_mm:                   {SIGMA*1e3:.1f}",
        f"z_m_mm:                     {Z_M*1e3:.0f}",
        f"f0_MHz:                     {F0/1e6:.1f}",
        f"grid:                       {NX}x{NY}  dx={DX*1e3:.3f}mm",
    ])

    if not passed and not args.allow_failure:
        sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
