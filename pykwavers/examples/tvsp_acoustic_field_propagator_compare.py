#!/usr/bin/env python3
"""
Validation of pykwavers ``angular_spectrum_cw`` against the second
Rayleigh-Sommerfeld (RS-2) integral for a pressure-specified aperture source.

Physical setup
--------------
A circular piston of radius *a* = 2 mm carries a uniform CW pressure
amplitude P₀ = 1 Pa at f₀ = 1 MHz in a lossless medium (c₀ = 1500 m/s).
The near-field transition distance is z_n = a² / λ = a²·f₀/c₀ ≈ 2.67 mm.

Two independent computations of the on-axis pressure profile from z = 5 mm
(just past z_n) to z = 50 mm are compared:

1. **pykwavers ``angular_spectrum_cw``** — spectral propagator:

       H(kx,ky,z) = conj(exp(j·kz·z)),  kz = sqrt(k² − kx² − ky²)

2. **Numerical RS-2 integral** — direct quadrature of the Sommerfeld formula:

       P(0,z) = z/(2π) · ∫∫_disc P₀ · (jk − 1/R) / R² · exp(−jkR) · dA

   where R = sqrt(x'² + y'² + z²) for on-axis observation.

Both compute the Helmholtz half-space field for a Dirichlet (pressure-specified)
boundary condition. Their finite-grid disagreement is limited by FFT truncation
and quadrature discretisation, not by a physics mismatch.

Pass criterion
--------------
Pearson r ≥ 0.99 and PSNR ≥ 30 dB on the on-axis |pressure| profile.
The tolerance is set by discrete near-field numerical error (RS-2 quadrature
and ASM FFT aliasing) and not by the comparison algorithm itself.
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
DX = 0.10e-3            # grid spacing [m]  (λ/15 at 1 MHz)
NX = NY = 128           # source plane grid size
APERTURE_RADIUS = 2e-3  # piston radius [m]; z_n = a²·f₀/c₀ ≈ 2.67 mm
P0 = 1.0                # source pressure amplitude [Pa]

# Propagation range: start past z_n to avoid extreme near-field
Z_START = 5e-3          # [m]
Z_END   = 50e-3         # [m]
N_Z     = 100

PEARSON_THRESHOLD = 0.990
PSNR_THRESHOLD_DB = 30.0

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "tvsp_acoustic_field_propagator_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_acoustic_field_propagator_metrics.txt"


def _circular_piston_source() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (source plane, disc x-coords, disc y-coords)."""
    x = (np.arange(NX) - NX / 2.0) * DX
    y = (np.arange(NY) - NY / 2.0) * DX
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = xx**2 + yy**2 <= APERTURE_RADIUS**2
    src = np.zeros((NX, NY), dtype=complex)
    src[mask] = P0
    return src, xx[mask].ravel(), yy[mask].ravel()


def rs2_on_axis(z_arr: np.ndarray, x_disc: np.ndarray, y_disc: np.ndarray) -> np.ndarray:
    """Numerical RS-2 integral (pressure-specified, Sommerfeld formula).

    P(0,z) = z/(2π) · ∫∫ P₀ · (jk − 1/R)/R² · exp(−jkR) · dA

    The jk/R² term dominates for kR >> 1 (satisfied for z ≥ Z_START = 5 mm:
    kR_min = 4189 × 5×10⁻³ = 21 >> 1).  The full two-term expression is used
    for accuracy.
    """
    k = 2.0 * np.pi * F0 / C0
    dA = DX**2
    result = np.empty(len(z_arr), dtype=complex)
    for i, z in enumerate(z_arr):
        R = np.sqrt(x_disc**2 + y_disc**2 + z**2)
        integrand = (1j * k - 1.0 / R) / R**2 * np.exp(-1j * k * R) * dA
        result[i] = (z / (2.0 * np.pi)) * P0 * integrand.sum()
    return result


def _psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref_max = float(ref.max())
    if ref_max == 0:
        return float("inf")
    mse = float(np.mean((ref - test) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(ref_max**2 / mse)


def save_figure(z_arr: np.ndarray, asm: np.ndarray, rs2: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    z_mm = z_arr * 1e3

    axes[0].plot(z_mm, np.abs(rs2), label="RS-2 numerical (reference)", lw=2)
    axes[0].plot(z_mm, np.abs(asm), "--", label="pykwavers ASM", lw=1.5)
    axes[0].set_xlabel("z [mm]")
    axes[0].set_ylabel("|P| [Pa]")
    axes[0].set_title("On-axis |pressure|")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    diff = np.abs(np.abs(asm) - np.abs(rs2))
    axes[1].plot(z_mm, diff, color="tab:red")
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("||P_ASM| − |P_RS2|| [Pa]")
    axes[1].set_title("Absolute error")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"tvsp_acoustic_field_propagator: circular piston a={APERTURE_RADIUS*1e3:.0f}mm"
        f"  f₀={F0/1e6:.1f}MHz  c₀={C0:.0f}m/s"
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

    src, x_disc, y_disc = _circular_piston_source()
    z_arr = np.linspace(Z_START, Z_END, N_Z)
    k = 2.0 * np.pi * F0 / C0
    z_n = APERTURE_RADIUS**2 * F0 / C0

    print("Computing pykwavers ASM on-axis profile...")
    pressure = pkw.angular_spectrum_cw(
        input_plane=src.real,
        dx=DX,
        z_pos=z_arr.tolist(),
        f0=F0,
        medium=C0,
        angular_restriction=True,
    )
    cx, cy = NX // 2, NY // 2
    asm_on_axis = pressure[cx, cy, :]

    print("Computing RS-2 reference integral...")
    rs2_on_axis_arr = rs2_on_axis(z_arr, x_disc, y_disc)

    asm_abs = np.abs(asm_on_axis)
    rs2_abs = np.abs(rs2_on_axis_arr)

    pearson_r, _ = pearsonr(rs2_abs, asm_abs)
    psnr_db = _psnr(rs2_abs, asm_abs)
    rms_ratio = float(np.sqrt(np.mean(asm_abs**2)) / np.sqrt(np.mean(rs2_abs**2))) if rs2_abs.any() else 1.0
    max_abs_err = float(np.max(np.abs(asm_abs - rs2_abs)))

    status = "PASS" if pearson_r >= PEARSON_THRESHOLD and psnr_db >= PSNR_THRESHOLD_DB else "FAIL"

    report_lines = [
        "tvsp_acoustic_field_propagator parity report",
        "============================================",
        "",
        f"Source: circular piston, radius a={APERTURE_RADIUS*1e3:.0f} mm",
        f"Frequency: f₀={F0/1e6:.1f} MHz,  k={k:.1f} rad/m",
        f"Medium: c₀={C0:.0f} m/s (lossless)",
        f"Grid: {NX}×{NY},  dx={DX*1e3:.3f} mm",
        f"Near-field transition: z_n = a²·f₀/c₀ = {z_n*1e3:.2f} mm",
        f"z range: [{Z_START*1e3:.1f}, {Z_END*1e3:.1f}] mm  ({N_Z} planes)",
        "",
        "Reference: numerical RS-2 integral (pressure-specified Sommerfeld formula)",
        "  P(0,z) = z/(2π) · ∫∫_disc (jk − 1/R)/R² · exp(−jkR) · dA",
        "",
        "On-axis |pressure| parity metrics",
        "----------------------------------",
        f"  pearson_r:   {pearson_r:.6f}  (threshold ≥ {PEARSON_THRESHOLD})",
        f"  psnr_db:     {psnr_db:.2f} dB  (threshold ≥ {PSNR_THRESHOLD_DB} dB)",
        f"  rms_ratio:   {rms_ratio:.6f}",
        f"  max_abs_err: {max_abs_err:.4e} Pa",
        "",
        f"parity_status: {status}",
    ]

    for line in report_lines:
        print(line)

    save_figure(z_arr, asm_on_axis, rs2_on_axis_arr, FIGURE_PATH)
    save_text_report(
        METRICS_PATH, "tvsp_acoustic_field_propagator parity report", report_lines
    )
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {METRICS_PATH}")

    if status == "FAIL" and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
