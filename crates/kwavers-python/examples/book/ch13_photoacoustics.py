"""
Chapter 13 figure generation — Photoacoustics
==============================================

Produces publication-quality figures for docs/book/photoacoustics.md.
Figures derive from closed-form analytical expressions for photoacoustic
signal generation and detection.

Output directory: docs/book/figures/ch13/

Figures produced
----------------
fig01  Optical absorption spectra of HbO₂ and Hb (Prahl 1999 polynomial fits)
fig02  Grüneisen parameter Γ vs temperature for water (Sigrist & Kneubühl 1978)
fig03  Photoacoustic signal from a spherical absorber (N-wave, Xu & Wang 2006)
fig04  PA axial resolution δz vs transducer bandwidth (Xu & Wang 2006)
fig05  Spectroscopic unmixing: two-chromophore least-squares solution

References
----------
Xu & Wang (2006) Rev. Sci. Instrum. 77:041101
Beard (2011) Interface Focus 1:602
Jacques (2013) Phys. Med. Biol. 58:R37
Prahl (1999) https://omlc.org/spectra/hemoglobin/
Sigrist & Kneubühl (1978) J. Acoust. Soc. Am. 64:1652
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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


# ── Figure 01: Optical absorption spectra ─────────────────────────────────────
def fig01_absorption_spectra() -> None:
    """
    Molar absorption coefficients ε(λ) [m⁻¹/(mol/L)]:
    HbO₂ and Hb: kw.hbo2_molar_absorption / kw.hb_molar_absorption
        6th-order polynomial fits to Prahl (1999) tabulated data, λ ∈ [650, 1000] nm.
    Melanin: μ_a ∝ λ^{-3.48} (Jacques 2013) — no Rust binding; kept as Python.
    Water: simplified near-IR model — no Rust binding; kept as Python.
    All curves normalised to [0, 1] for direct comparison.
    """
    lam_nm = np.linspace(650, 1000, 500)

    HbO2 = np.asarray(kw.hbo2_molar_absorption(lam_nm))
    Hb = np.asarray(kw.hb_molar_absorption(lam_nm))
    # Normalise to [0, 1] for visualisation
    HbO2 = HbO2 / (HbO2.max() + 1e-30)
    Hb = Hb / (Hb.max() + 1e-30)

    # Melanin: power-law decrease (Jacques 2013) — no kw binding
    melanin = (lam_nm / 650.0) ** (-3.48)
    melanin /= melanin.max()

    # Water: low in NIR window, rising above 900 nm (simplified) — no kw binding
    water = 0.05 + 0.3 * ((lam_nm - 650) / 350) ** 3
    water /= water.max()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lam_nm, HbO2, label=r"$\mathrm{HbO_2}$ (Prahl 1999 poly fit)")
    ax.plot(lam_nm, Hb, "--", label=r"$\mathrm{Hb}$ (Prahl 1999 poly fit)")
    ax.plot(lam_nm, melanin, "-.", label=r"Melanin $\propto\lambda^{-3.48}$ (Jacques 2013)")
    ax.plot(lam_nm, water, ":", label="Water (simplified NIR model)")
    ax.axvline(800, color="k", linewidth=0.5, linestyle="--", label="Isosbestic point ~800 nm")
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel("Normalised molar absorption (a.u.)")
    ax.set_title("Near-IR optical absorption spectra\n(kw.hbo2_molar_absorption / kw.hb_molar_absorption)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(650, 1000)
    fig.tight_layout()
    savefig("fig01_absorption_spectra")
    plt.close(fig)


# ── Figure 02: Grüneisen parameter vs temperature ────────────────────────────
def fig02_gruneisen_temperature() -> None:
    """
    Grüneisen parameter of water (Sigrist & Kneubühl 1978):
      Γ(T) = 0.0043 + 0.0053·T    (valid 0–60 °C)
    Computed via kw.gruneisen_parameter_water(t_celsius_arr) (Rust kernel).
    Γ = 0 at T ≈ −0.81 °C (where thermal expansion vanishes).
    At 37 °C: Γ ≈ 0.200.
    """
    T = np.linspace(0, 80, 300)
    Gamma = np.asarray(kw.gruneisen_parameter_water(T))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T, Gamma, color="#1f77b4")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(37, color="r", linestyle="--", linewidth=1, label="Body temp 37°C")
    ax.scatter([37], [float(np.asarray(kw.gruneisen_parameter_water(np.array([37.0])))[0])],
               s=60, color="r", zorder=5)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Grüneisen parameter $\Gamma$")
    ax.set_title(r"Grüneisen parameter vs temperature (water)"
                 "\n" r"$\Gamma(T) = 0.0043 + 0.0053\,T$ (Sigrist \& Kneubühl 1978)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig02_gruneisen_temperature")
    plt.close(fig)


# ── Figure 03: Photoacoustic signal from spherical absorber ───────────────────
def fig03_pa_sphere_signal() -> None:
    """
    Far-field N-wave photoacoustic signal from a uniformly heated sphere.
    Computed via kw.pa_sphere_pressure_signal(t_arr, r0_m, gamma, mua_per_m,
        c, r_det_m, initial_pressure_pa).
    initial_pressure_pa = Γ · μ_a · Φ  [Pa] — absorbed energy rise.
    Xu & Wang (2006) Rev. Sci. Instrum. 77:041101, eq. (13).
    """
    R = 0.001    # 1 mm sphere radius
    r_d = 0.05  # 50 mm detector distance
    c = 1500.0
    Gamma = 0.2
    mu_a = 100.0   # m⁻¹ absorption coefficient
    Phi = 1.0      # J/m² fluence
    # Initial pressure rise: p₀ = Γ · μ_a · Φ [Pa]
    p0 = Gamma * mu_a * Phi   # 20 Pa

    t_arr = np.linspace(0, 80e-6, 3000)  # 80 µs window
    p = np.asarray(kw.pa_sphere_pressure_signal(t_arr, R, Gamma, mu_a, c, r_d, p0))

    t1 = (r_d - R) / c   # leading edge arrival [s]
    t2 = (r_d + R) / c   # trailing edge arrival [s]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_arr * 1e6, p, color="#1f77b4")
    ax.axvline(t1 * 1e6, color="gray", linestyle="--", linewidth=1, label=f"$t_1={t1*1e6:.1f}$ µs")
    ax.axvline(t2 * 1e6, color="gray", linestyle=":", linewidth=1, label=f"$t_2={t2*1e6:.1f}$ µs")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel(r"Time $t$ (µs)")
    ax.set_ylabel("Photoacoustic pressure (Pa)")
    ax.set_title(f"PA signal from spherical absorber ($R={R*1e3:.1f}$ mm, $r_d={r_d*1e3:.0f}$ mm)\n"
                 r"(kw.pa_sphere_pressure_signal, Xu \& Wang 2006)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig03_pa_sphere_signal")
    plt.close(fig)


# ── Figure 04: PA axial resolution vs transducer bandwidth ────────────────────
def fig04_bandwidth_vs_radius() -> None:
    """
    PA axial resolution: δz = c / (2 · BW)
    Computed via kw.pa_axial_resolution(bandwidth_hz, c) (Rust kernel).
    Equivalently: minimum detectable sphere radius R ≈ δz / 2 for transducer BW.
    Xu & Wang (2006) Rev. Sci. Instrum. 77:041101.
    """
    c = 1500.0
    B_Hz = np.logspace(5, 9, 300)   # 0.1 MHz to 1 GHz
    dz_mm = np.array([kw.pa_axial_resolution(b, c) * 1e3 for b in B_Hz])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(B_Hz * 1e-6, dz_mm, color="#1f77b4")

    for f_MHz, col in [(1.0, "#aaa"), (5.0, "#888"), (20.0, "#666"), (100.0, "#444")]:
        dz = kw.pa_axial_resolution(f_MHz * 1e6, c) * 1e3  # mm
        ax.axvline(f_MHz, color=col, linestyle="--", linewidth=0.8)
        ax.axhline(dz, color=col, linestyle=":", linewidth=0.8)
        ax.text(f_MHz * 1.15, dz * 0.7, f"{f_MHz:.0f} MHz\n→ {dz:.2f} mm",
                fontsize=8, color=col, va="top")

    ax.set_xlabel("Transducer bandwidth (MHz)")
    ax.set_ylabel(r"Axial resolution $\delta z$ (mm)")
    ax.set_title(r"PA axial resolution: $\delta z = c\,/(2\,\mathrm{BW})$"
                 "\n(kw.pa_axial_resolution)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(B_Hz[0] * 1e-6, B_Hz[-1] * 1e-6)
    fig.tight_layout()
    savefig("fig04_bandwidth_vs_radius")
    plt.close(fig)


# ── Figure 05: Spectroscopic unmixing (two chromophores) ──────────────────────
def fig05_spectroscopic_unmixing() -> None:
    """
    At N wavelengths, PA amplitude p(λ) = Σ_k ε_k(λ) · c_k · Γ · Φ.
    Two chromophores (HbO₂, Hb) at 760/850 nm:
      ε from kw.hbo2_molar_absorption / kw.hb_molar_absorption (Prahl 1999).
    Unmixing via kw.spectroscopic_unmixing_lstsq(spectra_matrix, measurements)
      which solves (AᵀA)x = Aᵀb by Gaussian elimination (Beard 2011).
    Plot: sO₂ estimate vs true sO₂ for deterministic measurement perturbations.
    """
    lam_unmix = np.array([760.0, 850.0])
    perturbations = np.array([0.0, 0.02, 0.05])
    sweep = kw.spectroscopic_unmixing_so2_sweep(
        lam_unmix, 0.0, 1.0, 100, perturbations
    )
    sO2_true = np.asarray(sweep["true_so2"])
    sO2_estimates = np.asarray(sweep["estimated_so2_by_perturbation"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["No perturbation", "2% perturbation", "5% perturbation"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for estimate, col, lbl in zip(sO2_estimates, colors, labels):
        ax.plot(sO2_true, estimate, color=col, label=lbl)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel(r"True $sO_2$")
    ax.set_ylabel(r"Estimated $sO_2$")
    ax.set_title("Spectroscopic unmixing: HbO₂/Hb at 760/850 nm\n"
                 "(kw.spectroscopic_unmixing_so2_sweep, Beard 2011)")
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
