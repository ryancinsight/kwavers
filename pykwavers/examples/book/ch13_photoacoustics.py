"""
Chapter 13 figure generation — Photoacoustics
==============================================

Produces publication-quality figures for docs/book/photoacoustics.md.
Figures derive from closed-form analytical expressions for photoacoustic
signal generation and detection.

Output directory: docs/book/figures/ch13/

Figures produced
----------------
fig01  Optical absorption spectra of HbO₂, Hb, melanin, water (model)
fig02  Grüneisen parameter Γ vs temperature for water/tissue
fig03  Photoacoustic signal from a spherical absorber (analytical)
fig04  PA signal bandwidth vs absorber radius (stress confinement)
fig05  Spectroscopic unmixing: two-chromophore least-squares solution

References
----------
Xu & Wang (2006) Rev. Mod. Phys. 78:1338
Beard (2011) Interface 8(54):1271
Jacques (2013) Phys. Med. Biol. 58:R37
Ntziachristos (2010) Nat. Methods 7:603
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch13")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch13/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})


# ── Figure 01: Optical absorption spectra (analytical models) ─────────────────
def fig01_absorption_spectra() -> None:
    """
    Molar extinction coefficients (cm⁻¹/M):
    HbO₂ and Hb: Prahl table simplified as Gaussian + exponential models
    Melanin: μ_a ∝ λ^{-3.48} (Jacques 2013)
    Water: μ_a absorption coefficient (cm⁻¹), simplified near-IR model.
    All in arbitrary normalised units for illustration.
    """
    lam_nm = np.linspace(650, 1000, 500)  # near-IR window

    # Simplified analytical models (not tabulated data)
    # HbO2: main peak near 920 nm + shoulder at 760 nm (normalised)
    HbO2 = (0.5 * np.exp(-((lam_nm - 920) / 30)**2)
             + 0.3 * np.exp(-((lam_nm - 760) / 20)**2))

    # Hb: main peak near 760 nm (isosbestic at ~800 nm)
    Hb = (0.9 * np.exp(-((lam_nm - 760) / 25)**2)
          + 0.1 * np.exp(-((lam_nm - 900) / 40)**2))

    # Melanin: power-law decrease
    melanin = 2.0 * (lam_nm / 650.0)**(-3.48)
    melanin /= melanin.max()

    # Water: low in NIR window, rising at >900 nm
    water = 0.05 + 0.3 * ((lam_nm - 650) / 350)**3

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lam_nm, HbO2, label=r"$\mathrm{HbO_2}$ (oxyhaemoglobin)")
    ax.plot(lam_nm, Hb, "--", label=r"$\mathrm{Hb}$ (deoxyhaemoglobin)")
    ax.plot(lam_nm, melanin, "-.", label="Melanin")
    ax.plot(lam_nm, water, ":", label=r"Water ($\times 0.1$)")
    ax.axvline(800, color="k", linewidth=0.5, linestyle="--", label="Isosbestic point ~800 nm")
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel("Normalised absorption (a.u.)")
    ax.set_title("Near-IR optical absorption spectra (analytical models)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(650, 1000)
    fig.tight_layout()
    savefig("fig01_absorption_spectra")
    plt.close(fig)


# ── Figure 02: Grüneisen parameter vs temperature ────────────────────────────
def fig02_gruneisen_temperature() -> None:
    """
    Γ = β c² / Cp
    For water: β(T) = -0.088 + 0.0122T - 1.5e-4 T² + 9e-7 T³  (10⁻⁴ K⁻¹)
               c(T) = 1402.7 + 4.88T - 0.048T²  (m/s)
               Cp = 4182 J/(kg·K)  (weakly temperature-dependent)
    Gamma has a minimum near 4 °C where β = 0.
    """
    T = np.linspace(0, 80, 300)
    # β (thermal expansion, 10⁻⁴ K⁻¹) — simplified polynomial
    beta = (-0.0882 + 0.0122 * T - 1.54e-4 * T**2 + 9.0e-7 * T**3) * 1e-4  # K⁻¹
    c = 1402.7 + 4.88 * T - 0.048 * T**2  # m/s
    Cp = 4182.0  # J/(kg·K)
    Gamma = beta * c**2 / Cp

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T, Gamma, color="#1f77b4")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(37, color="r", linestyle="--", linewidth=1, label="Body temp 37°C")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Grüneisen parameter $\Gamma = \beta c^2 / C_p$")
    ax.set_title(r"Grüneisen parameter $\Gamma$ vs temperature (water)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig02_gruneisen_temperature")
    plt.close(fig)


# ── Figure 03: Photoacoustic signal from spherical absorber ───────────────────
def fig03_pa_sphere_signal() -> None:
    """
    Exact photoacoustic signal from a uniformly heated sphere of radius R:
    p(r, t) = (Γ μ_a Φ / 2) · [r - |r-t c|] / r  for |r-ct| ≤ R
    Time-domain pressure at detector distance r_d:
    p(t) = (Γ μ_a Φ R³) / (3 r_d c) · d/dt [N-wave shape]
    For detector at r_d >> R:
    p(t) = (Γ H_0 c / 2) · (t-t1)/(t2-t1) for t1 < t < t2 (N-wave)
    Using Xu & Wang 2006 Eq. 2.
    """
    R = 0.001    # 1 mm sphere radius
    r_d = 0.05  # 50 mm detector distance
    c = 1500.0
    Gamma = 0.2
    mu_a = 100.0   # m⁻¹ absorption coefficient
    Phi = 1.0      # J/m² fluence (normalised)

    t_arr = np.linspace(0, 80e-6, 3000)  # 80 µs window
    t1 = (r_d - R) / c   # arrival of leading edge
    t2 = (r_d + R) / c   # arrival of trailing edge

    # Analytical N-wave (far-field approximation, Xu & Wang 2006 Eq.8)
    p = np.zeros_like(t_arr)
    mask = (t_arr >= t1) & (t_arr <= t2)
    # N-wave: pressure proportional to time derivative of spherical wave
    t_centre = (t1 + t2) / 2
    T_half = (t2 - t1) / 2
    # Normalised N-wave shape: (t - t_centre) / T_half
    amplitude = Gamma * mu_a * Phi * c * R / (2 * r_d)
    p[mask] = amplitude * (t_arr[mask] - t_centre) / T_half

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_arr * 1e6, p, color="#1f77b4")
    ax.axvline(t1 * 1e6, color="gray", linestyle="--", linewidth=1, label=f"$t_1={t1*1e6:.1f}$ µs")
    ax.axvline(t2 * 1e6, color="gray", linestyle=":", linewidth=1, label=f"$t_2={t2*1e6:.1f}$ µs")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel(r"Time $t$ (µs)")
    ax.set_ylabel("Photoacoustic pressure (a.u.)")
    ax.set_title(f"PA signal from spherical absorber ($R={R*1e3:.1f}$ mm, $r_d={r_d*1e3:.0f}$ mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig03_pa_sphere_signal")
    plt.close(fig)


# ── Figure 04: Bandwidth vs absorber radius (stress confinement) ──────────────
def fig04_bandwidth_vs_radius() -> None:
    """
    Stress confinement requires τ_L >> τ_stress = R/c.
    Signal bandwidth B ≈ c/(2R) — half-power bandwidth of N-wave spectrum.
    For detection: need transducer BW > c/(2R).
    """
    R = np.logspace(-4, -2, 200)  # 0.1 mm to 10 mm
    c = 1500.0
    B_Hz = c / (2 * R)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(R * 1e3, B_Hz * 1e-6, color="#1f77b4")
    ax.set_xlabel("Absorber radius $R$ (mm)")
    ax.set_ylabel("Signal bandwidth $B$ (MHz)")
    ax.set_title(r"PA signal bandwidth: $B \approx c/(2R)$")
    # Annotate typical transducer bandwidths
    for f_MHz, lbl in [(1, "1 MHz"), (5, "5 MHz"), (20, "20 MHz"), (100, "100 MHz")]:
        R_at = c / (2 * f_MHz * 1e6) * 1e3  # mm
        ax.axhline(f_MHz, color="gray", linestyle="--", linewidth=0.8)
        ax.text(0.12, f_MHz * 1.1, lbl, fontsize=8, color="gray")

    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.05, 200)
    fig.tight_layout()
    savefig("fig04_bandwidth_vs_radius")
    plt.close(fig)


# ── Figure 05: Spectroscopic unmixing (two chromophores) ──────────────────────
def fig05_spectroscopic_unmixing() -> None:
    """
    At N wavelengths, measured PA amplitude p(λ) = Σ_k ε_k(λ) c_k · Γ Φ(λ).
    Two chromophores (HbO₂, Hb): solve 2×2 system at two wavelengths
    and show how sO₂ = c_HbO2 / (c_HbO2 + c_Hb) varies vs measured ratios.
    Plot: sO₂ estimate vs true sO₂ for different noise levels.
    """
    # Molar extinction at 760 nm and 850 nm (simplified, relative units)
    eps_HbO2 = np.array([0.30, 0.80])  # [760nm, 850nm]
    eps_Hb = np.array([0.85, 0.30])    # [760nm, 850nm]

    # True sO₂ sweep
    sO2_true = np.linspace(0.0, 1.0, 100)
    c_HbO2 = sO2_true
    c_Hb = 1.0 - sO2_true

    # Measured PA signal (noiseless)
    E = np.column_stack([eps_HbO2, eps_Hb])  # 2×2 matrix [wavelength × chrom]
    C_true = np.column_stack([c_HbO2, c_Hb])  # 100×2

    # sO₂ estimate from least-squares inversion
    E_inv = np.linalg.pinv(E)  # 2×2 pseudo-inverse

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for noise_level, col, lbl in [(0.0, "#1f77b4", "No noise"),
                                   (0.02, "#ff7f0e", "2% noise"),
                                   (0.05, "#2ca02c", "5% noise")]:
        rng = np.random.default_rng(42)
        sO2_est_all = []
        for i, sO2 in enumerate(sO2_true):
            PA = E @ C_true[i]  # 2-wavelength measurement
            PA_noisy = PA + noise_level * rng.standard_normal(2) * PA.max()
            c_est = E_inv @ PA_noisy
            c_est = np.clip(c_est, 0, None)
            s = c_est[0] / (c_est.sum() + 1e-12)
            sO2_est_all.append(s)
        ax.plot(sO2_true, sO2_est_all, color=col, label=lbl)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel(r"True $sO_2$")
    ax.set_ylabel(r"Estimated $sO_2$")
    ax.set_title("Spectroscopic unmixing: HbO₂/Hb at 760/850 nm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    savefig("fig05_spectroscopic_unmixing")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 13 figures (Photoacoustics)...")
    fig01_absorption_spectra()
    fig02_gruneisen_temperature()
    fig03_pa_sphere_signal()
    fig04_bandwidth_vs_radius()
    fig05_spectroscopic_unmixing()
    print("Done. Output: docs/book/figures/ch13/")
