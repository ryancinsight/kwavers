#!/usr/bin/env python3
"""
pr_2D_attenuation_compensation_compare.py
==========================================
Validates frequency-domain attenuation compensation for CW angular spectrum
propagation in an absorbing medium.

Physical setup
--------------
A 2-D Gaussian source (σ=3 mm, P₀=1 Pa, f₀=1 MHz) propagated to z_m=30 mm
in a lossy medium (α₀=3 dB/MHz/cm, y=1) using ``angular_spectrum_cw``.

One-way attenuation at f₀, z_m:
    Δ = α₀ [dB/MHz/cm] × f₀ [MHz] × z_m [cm] = 3 × 1 × 3 = 9 dB

Compensation scheme
-------------------
The CW angular spectrum applies absorption Eq. 11 of Zeng & McGough (2008):
    H_abs(kx,ky) = exp(−α_Np · z_m · k / kz)
where α_Np [Np/m] = α₀ × 100 / 8.686 × (f₀/1e6)^y.

For on-axis propagating waves (kz ≈ k):
    H_abs ≈ exp(−α_Np · z_m)

The compensation filter applied to the measured field:
    H_comp = exp(+α_Np · z_m)   [scalar, on-axis approximation]

restores the lossless field amplitude for propagating components.

Parity criterion
----------------
  PSNR(source, backward(compensated measured)) ≥ PSNR(source, backward(raw measured)) + 3 dB
  PSNR(source, backward(compensated measured)) ≥ 45 dB

Outputs
-------
  output/pr_2D_attenuation_compensation_compare.png
  output/pr_2D_attenuation_compensation_metrics.txt
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
F0 = 1_000_000.0    # [Hz]
C0 = 1500.0         # [m/s]
DX = 0.15e-3        # [m]  λ/10 at 1 MHz
NX = NY = 128
SIGMA = 3e-3        # [m]  Gaussian half-width
P0 = 1.0            # [Pa]
Z_M = 30e-3         # [m]  measurement plane distance

ALPHA0_DB_MHZ_CM = 3.0   # [dB/(MHz·cm)]  → 9 dB one-way at f₀, z_m
ALPHA_POWER      = 1.0

# One-way α at f₀ in Np/m
ALPHA_NP = ALPHA0_DB_MHZ_CM * 100.0 / (20.0 / np.log(10.0)) * (F0 / 1.0e6) ** ALPHA_POWER

# Parity thresholds
TOL_PSNR_COMP    = 45.0   # [dB] minimum PSNR after compensation
TOL_PSNR_GAIN_DB = 3.0    # [dB] PSNR must improve by at least this

FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "pr_2D_attenuation_compensation_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "pr_2D_attenuation_compensation_metrics.txt"


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref_max = float(np.abs(ref).max())
    if ref_max == 0:
        return float("inf")
    mse = float(np.mean(np.abs(ref - test) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(ref_max ** 2 / mse)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    attn_dB = ALPHA0_DB_MHZ_CM * (F0 / 1e6) * Z_M * 100.0
    print(
        f"pr_2D_attenuation_compensation (CW angular spectrum):\n"
        f"  f₀={F0/1e6:.1f} MHz, σ={SIGMA*1e3:.1f} mm, z_m={Z_M*1e3:.0f} mm\n"
        f"  α₀={ALPHA0_DB_MHZ_CM} dB/MHz/cm  → one-way {attn_dB:.1f} dB at f₀"
    )

    # Source field
    src = pkw.gaussian_source_2d(NX, NY, DX, SIGMA, amplitude=P0)

    # Lossless medium
    med_lossless = C0

    # Absorbing medium (dict form for angular_spectrum_cw)
    med_absorbing = {
        "sound_speed": C0,
        "alpha_coeff": ALPHA0_DB_MHZ_CM,
        "alpha_power": ALPHA_POWER,
    }

    # --- Forward propagation (lossless) ---
    p_fwd_lossless = pkw.angular_spectrum_cw(
        src.real, DX, [Z_M], F0, med_lossless, angular_restriction=True,
    )[:, :, 0]   # (NX, NY)

    # --- Forward propagation (absorbing) ---
    p_fwd_absorb = pkw.angular_spectrum_cw(
        src.real, DX, [Z_M], F0, med_absorbing, angular_restriction=True,
    )[:, :, 0]   # (NX, NY)

    # --- Apply on-axis amplitude compensation to absorbing measurement ---
    #   H_comp ≈ exp(+α_Np · z_m)   (scalar for on-axis components)
    p_fwd_comp = p_fwd_absorb * np.exp(ALPHA_NP * Z_M)

    # --- Backward reconstruction from each measurement ---
    p_recon_raw  = pkw.backward_angular_spectrum_cw(
        p_fwd_absorb, DX, Z_M, F0, med_lossless, angular_restriction=True,
    )
    p_recon_comp = pkw.backward_angular_spectrum_cw(
        p_fwd_comp,   DX, Z_M, F0, med_lossless, angular_restriction=True,
    )
    p_recon_loss = pkw.backward_angular_spectrum_cw(
        p_fwd_lossless, DX, Z_M, F0, med_lossless, angular_restriction=True,
    )

    src_abs       = np.abs(src)
    recon_raw_abs = np.abs(p_recon_raw)
    recon_comp_abs= np.abs(p_recon_comp)
    recon_loss_abs= np.abs(p_recon_loss)

    psnr_lossless = _psnr(src_abs, recon_loss_abs)
    psnr_raw      = _psnr(src_abs, recon_raw_abs)
    psnr_comp     = _psnr(src_abs, recon_comp_abs)
    psnr_gain     = psnr_comp - psnr_raw

    passed = psnr_comp >= TOL_PSNR_COMP and psnr_gain >= TOL_PSNR_GAIN_DB
    status = "PASS" if passed else "FAIL"

    print(f"\n  2D attenuation compensation [{status}]:")
    print(f"    PSNR (lossless recon vs src):  {psnr_lossless:.2f} dB")
    print(f"    PSNR (raw absorb recon vs src): {psnr_raw:.2f} dB")
    print(f"    PSNR (comp recon vs src):       {psnr_comp:.2f} dB  (threshold ≥ {TOL_PSNR_COMP})")
    print(f"    PSNR gain:                      {psnr_gain:+.2f} dB  (threshold ≥ {TOL_PSNR_GAIN_DB})")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    kw = dict(origin="lower", cmap="viridis")

    axes[0, 0].imshow(src_abs.T, **kw)
    axes[0, 0].set_title(f"Source |P| (z=0, σ={SIGMA*1e3:.0f}mm)")

    axes[0, 1].imshow(np.abs(p_fwd_lossless).T, **kw)
    axes[0, 1].set_title(f"Lossless |P| (z={Z_M*1e3:.0f}mm)")

    axes[0, 2].imshow(np.abs(p_fwd_absorb).T, **kw)
    axes[0, 2].set_title(f"Absorbing |P| (z={Z_M*1e3:.0f}mm, Δ={attn_dB:.0f}dB)")

    axes[1, 0].imshow(recon_loss_abs.T, **kw)
    axes[1, 0].set_title(f"Recon (lossless) PSNR={psnr_lossless:.1f}dB")

    axes[1, 1].imshow(recon_raw_abs.T, **kw)
    axes[1, 1].set_title(f"Recon (no comp) PSNR={psnr_raw:.1f}dB")

    axes[1, 2].imshow(recon_comp_abs.T, **kw)
    axes[1, 2].set_title(f"Recon (compensated) PSNR={psnr_comp:.1f}dB [{status}]")

    for row in axes:
        for ax in row:
            ax.set_xlabel("x index"); ax.set_ylabel("y index")

    fig.suptitle(
        f"2D CW Attenuation Compensation [{status}]\n"
        f"f₀={F0/1e6:.1f}MHz  α₀={ALPHA0_DB_MHZ_CM}dB/MHz/cm  z={Z_M*1e3:.0f}mm  {attn_dB:.0f}dB one-way\n"
        f"PSNR gain={psnr_gain:+.2f}dB (≥{TOL_PSNR_GAIN_DB}dB)  PSNR_comp={psnr_comp:.1f}dB (≥{TOL_PSNR_COMP}dB)"
    )
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")
    print(f"parity_status: {status}")

    save_text_report(METRICS_PATH, "pr_2D_attenuation_compensation_compare", [
        f"Status:               {status}",
        f"psnr_lossless_db:     {psnr_lossless:.4f}",
        f"psnr_raw_db:          {psnr_raw:.4f}",
        f"psnr_comp_db:         {psnr_comp:.4f}  (threshold ≥ {TOL_PSNR_COMP})",
        f"psnr_gain_db:         {psnr_gain:.4f}  (threshold ≥ {TOL_PSNR_GAIN_DB})",
        f"attenuation_dB:       {attn_dB:.2f}",
        f"alpha0_db_mhz_cm:     {ALPHA0_DB_MHZ_CM}",
        f"alpha_power:          {ALPHA_POWER}",
        f"f0_MHz:               {F0/1e6:.1f}",
        f"sigma_mm:             {SIGMA*1e3:.1f}",
        f"z_m_mm:               {Z_M*1e3:.0f}",
        f"grid:                 {NX}x{NY}  dx={DX*1e3:.3f}mm",
    ])

    if not passed and not args.allow_failure:
        sys.exit(1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
