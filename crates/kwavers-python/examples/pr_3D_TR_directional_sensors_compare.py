#!/usr/bin/env python3
"""
pr_3D_TR_directional_sensors_compare.py
========================================
Validates 3D directional-sensor time-reversal reconstruction via CW angular
spectrum.  Tests the full roundtrip pipeline:

    source  →  forward ASM  →  noisy measurement
    noisy measurement  →  [directional filter × backward ASM]  →  reconstruction

The directional filter weight W(kx,ky) = (1+kz/k)/2 and the backward propagator
H_back(kx,ky) = exp(j·kz·z_m) are multiplied in a SINGLE padded N×N FFT step,
eliminating any intermediate spatial-domain truncation that would break the
linear equivalence.  This is the physically correct combined operation:

    recon_dir = IFFT_N(H_back × W × FFT_N(p_meas_noisy))[:NX, :NY]

compared to the omnidirectional baseline:

    recon_omni = IFFT_N(H_back × FFT_N(p_meas_noisy))[:NX, :NY]

Physical setup
--------------
A 2-D Gaussian CW source (σ=3 mm, f₀=1 MHz) propagated to a 2-D sensor plane
at z_m=20 mm.  Pixel-uniform in-band noise (krad ≤ k, 10 % RMS of signal RMS)
is added at the sensor plane.

Analytic noise power reduction (pixel-uniform forward-only in-band noise)
--------------------------------------------------------------------------
    E[|W|²] = 2∫₀^1[(1+√(1−u²))/2]² u du = 17/24
    → noise power reduced by −10·log10(17/24) ≈ 1.50 dB

The combined H_back × W operation preserves |H_back|² = 1 for propagating
waves, so the noise reduction factor is |W|² averaged over in-band pixels.

Parity criterion (reconstruction-domain PSNR comparison)
---------------------------------------------------------
  PSNR(src, recon_dir)   ≥ PSNR(src, recon_omni) + 0.8 dB
  PSNR(src, recon_dir)   ≥ 38 dB

Outputs
-------
  output/pr_3D_TR_directional_sensors_compare.png
  output/pr_3D_TR_directional_sensors_metrics.txt
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
NOISE_FRACTION = 0.10  # 10% RMS → PSNR_omni_recon ≈ 40 dB
RNG_SEED = 42

TOL_PSNR_DIR_DB  = 38.0   # minimum PSNR of directional reconstruction
TOL_PSNR_GAIN_DB = 0.8    # minimum gain; analytic ≈ 1.50 dB, measured ≈ 1.02 dB

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "pr_3D_TR_directional_sensors_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "pr_3D_TR_directional_sensors_metrics.txt"


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref_max = float(np.abs(ref).max())
    if ref_max == 0:
        return float("inf")
    mse = float(np.mean(np.abs(ref - test) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(ref_max ** 2 / mse)


def _padded_size(nx: int, ny: int) -> int:
    return int(2 ** (np.ceil(np.log2(max(nx, ny))) + 1))


def _backward_omni(
    p: np.ndarray, dx: float, z_m: float, f0: float, c0: float
) -> np.ndarray:
    """Backward propagation in padded N×N k-space, no directional filter."""
    k = 2.0 * np.pi * f0 / c0
    nx, ny = p.shape
    N = _padded_size(nx, ny)
    kx_vec = np.fft.fftfreq(N, d=dx / (2.0 * np.pi))
    ky_vec = np.fft.fftfreq(N, d=dx / (2.0 * np.pi))
    ky, kx = np.meshgrid(ky_vec, kx_vec, indexing="ij")
    krad = np.sqrt(kx**2 + ky**2)
    kz = np.sqrt((k**2 - kx**2 - ky**2).astype(complex))
    H_back = np.exp(1j * z_m * kz)
    H_back[krad > k] = 0.0   # suppress evanescent only
    return np.fft.ifft2(np.fft.fft2(p, (N, N)) * H_back)[:nx, :ny]


def _backward_directional(
    p: np.ndarray, dx: float, z_m: float, f0: float, c0: float
) -> np.ndarray:
    """Combined directional filter W + backward propagation H_back in single padded FFT.

    recon = IFFT_N(H_back × W × FFT_N(p))[:NX, :NY]
    No intermediate spatial-domain truncation — the two linear operations
    commute exactly within the same k-space grid.
    """
    k = 2.0 * np.pi * f0 / c0
    nx, ny = p.shape
    N = _padded_size(nx, ny)
    kx_vec = np.fft.fftfreq(N, d=dx / (2.0 * np.pi))
    ky_vec = np.fft.fftfreq(N, d=dx / (2.0 * np.pi))
    ky, kx = np.meshgrid(ky_vec, kx_vec, indexing="ij")
    krad = np.sqrt(kx**2 + ky**2)
    kz = np.sqrt((k**2 - kx**2 - ky**2).astype(complex))
    W = (1.0 + kz / k) / 2.0
    H_back = np.exp(1j * z_m * kz)
    H_back[krad > k] = 0.0   # suppress evanescent only
    return np.fft.ifft2(np.fft.fft2(p, (N, N)) * H_back * W)[:nx, :ny]


def _isotropic_inband_noise(
    nx: int, ny: int, dx: float, k: float, rms_target: float, rng
) -> np.ndarray:
    kx_vec = np.fft.fftfreq(nx, d=dx / (2.0 * np.pi))
    ky_vec = np.fft.fftfreq(ny, d=dx / (2.0 * np.pi))
    ky, kx = np.meshgrid(ky_vec, kx_vec, indexing="ij")
    in_band = np.sqrt(kx**2 + ky**2) <= k
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
    expected_gain_db = -10.0 * np.log10(17.0 / 24.0)
    N = _padded_size(NX, NY)
    print(
        f"pr_3D_TR_directional_sensors (combined H_back×W in single N×N FFT):\n"
        f"  Grid {NX}×{NY}, padded N={N}, dx={DX*1e3:.2f} mm\n"
        f"  f₀={F0/1e6:.1f} MHz, k={k:.0f} rad/m, z_m={Z_M*1e3:.0f} mm\n"
        f"  Noise={NOISE_FRACTION:.0%} RMS, analytically expected gain: {expected_gain_db:.2f} dB"
    )

    # --- Source and forward propagation ---
    src = pkw.gaussian_source_2d(NX, NY, DX, SIGMA, amplitude=P0)
    p_meas_clean = pkw.angular_spectrum_cw(
        src.real, DX, [Z_M], F0, C0, angular_restriction=True,
    )[:, :, 0]

    # --- Noise at measurement plane ---
    rng = np.random.default_rng(RNG_SEED)
    signal_rms = float(np.sqrt(np.mean(np.abs(p_meas_clean) ** 2)))
    noise_rms_target = NOISE_FRACTION * signal_rms
    noise = _isotropic_inband_noise(NX, NY, DX, k, noise_rms_target, rng)
    p_meas_noisy = p_meas_clean + noise

    # --- Lossless backward (for PSNR floor and visualization) ---
    recon_clean = _backward_omni(p_meas_clean, DX, Z_M, F0, C0)
    # --- Omnidirectional backward from noisy measurement ---
    recon_omni  = _backward_omni(p_meas_noisy, DX, Z_M, F0, C0)
    # --- Combined directional + backward from noisy measurement ---
    recon_dir   = _backward_directional(p_meas_noisy, DX, Z_M, F0, C0)

    src_abs = np.abs(src)
    psnr_lossless = _psnr(src_abs, np.abs(recon_clean))
    psnr_omni     = _psnr(src_abs, np.abs(recon_omni))
    psnr_dir      = _psnr(src_abs, np.abs(recon_dir))
    psnr_gain     = psnr_dir - psnr_omni

    passed = psnr_dir >= TOL_PSNR_DIR_DB and psnr_gain >= TOL_PSNR_GAIN_DB
    status = "PASS" if passed else "FAIL"

    print(f"\n  3D TR directional [{status}]:")
    print(f"    PSNR (lossless backward vs src):   {psnr_lossless:.2f} dB")
    print(f"    PSNR (omni noisy backward vs src): {psnr_omni:.2f} dB")
    print(f"    PSNR (dir backward vs src):        {psnr_dir:.2f} dB  (threshold ≥ {TOL_PSNR_DIR_DB})")
    print(f"    PSNR gain:                         {psnr_gain:+.2f} dB  (threshold ≥ {TOL_PSNR_GAIN_DB})")
    print(f"    Expected analytic gain:            {expected_gain_db:.2f} dB")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    kw = dict(origin="lower", cmap="viridis")

    axes[0, 0].imshow(src_abs.T, **kw)
    axes[0, 0].set_title(f"Source (z=0, σ={SIGMA*1e3:.0f}mm)")

    axes[0, 1].imshow(np.abs(p_meas_noisy).T, **kw)
    axes[0, 1].set_title(f"Omni sensor z={Z_M*1e3:.0f}mm\n(+{NOISE_FRACTION:.0%} RMS noise)")

    axes[0, 2].imshow(np.abs(recon_clean).T, **kw)
    axes[0, 2].set_title(f"Lossless backward PSNR={psnr_lossless:.1f}dB")

    axes[1, 0].imshow(np.abs(recon_omni).T, **kw)
    axes[1, 0].set_title(f"Omni backward PSNR={psnr_omni:.1f}dB")

    axes[1, 1].imshow(np.abs(recon_dir).T, **kw)
    axes[1, 1].set_title(
        f"Directional backward [{status}]\nPSNR={psnr_dir:.1f}dB  gain={psnr_gain:+.1f}dB"
    )

    diff = np.abs(np.abs(recon_dir) - np.abs(recon_omni))
    axes[1, 2].imshow(diff.T, **kw)
    axes[1, 2].set_title("|recon_dir| − |recon_omni|")

    for row in axes:
        for ax in row:
            ax.set_xlabel("x index"); ax.set_ylabel("y index")

    fig.suptitle(
        f"3D TR Directional Sensor [H_back×W combined] [{status}]\n"
        f"f₀={F0/1e6:.1f}MHz  σ={SIGMA*1e3:.0f}mm  z_m={Z_M*1e3:.0f}mm  noise={NOISE_FRACTION:.0%}RMS\n"
        f"PSNR gain={psnr_gain:+.2f}dB (≥{TOL_PSNR_GAIN_DB}dB, analytic≈{expected_gain_db:.1f}dB)  "
        f"PSNR_dir={psnr_dir:.1f}dB (≥{TOL_PSNR_DIR_DB}dB)"
    )
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")
    print(f"parity_status: {status}")

    save_text_report(METRICS_PATH, "pr_3D_TR_directional_sensors_compare", [
        f"Status:                      {status}",
        f"psnr_lossless_db:            {psnr_lossless:.4f}",
        f"psnr_omni_db:                {psnr_omni:.4f}",
        f"psnr_dir_db:                 {psnr_dir:.4f}  (threshold ≥ {TOL_PSNR_DIR_DB} dB)",
        f"psnr_gain_db:                {psnr_gain:.4f}  (threshold ≥ {TOL_PSNR_GAIN_DB})",
        f"expected_gain_analytic_db:   {expected_gain_db:.4f}",
        f"noise_fraction:              {NOISE_FRACTION}",
        f"noise_type:                  pixel-uniform in-band forward (krad<=k, NXxNY k-space)",
        f"backward_prop:               single padded N={N} FFT (H_back×W combined, no intermediate truncation)",
        f"sigma_mm:                    {SIGMA*1e3:.1f}",
        f"z_m_mm:                      {Z_M*1e3:.0f}",
        f"f0_MHz:                      {F0/1e6:.1f}",
        f"grid:                        {NX}x{NY}  dx={DX*1e3:.3f}mm",
    ])

    if not passed and not args.allow_failure:
        sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
