"""
Chapter 8 figure generation — Acoustic Propagation
===================================================

Produces publication-quality figures for docs/book/acoustic_propagation.md.
All figures derive from closed-form analytical expressions; no simulation
required (pykwavers optional for solver-comparison overlays).

Output directory: docs/book/figures/ch08/

Figures produced
----------------
fig01  Plane-wave propagation snapshots at four time instants
fig02  Spherical spreading: intensity decay 1/r² and 1/r (cylindrical)
fig03  Normal-incidence reflection/transmission coefficients at layered interfaces
fig04  Power-law absorption α(f) = α₀ f^y for six tissue types
fig05  Phase-velocity error: PSTD vs FDTD as a function of kΔx

References
----------
Kinsler et al. (2000) Fundamentals of Acoustics, 4th ed.
Treeby & Cox (2010) doi:10.1121/1.3377056
Duck (1990) Physical Properties of Tissue
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pykwavers as kw

# ── Output directory ─────────────────────────────────────────────────────────
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch08")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch08/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

# Physical constants
C0 = 1500.0    # m/s — water sound speed
F0 = 1.0e6     # Hz  — 1 MHz carrier
OMEGA0 = 2 * np.pi * F0
LAMBDA0 = C0 / F0   # 1.5 mm wavelength

# ── Figure 01: Plane-wave propagation snapshots ───────────────────────────────
def fig01_plane_wave_snapshots() -> None:
    # kw.plane_wave_pressure_1d(amplitude, k, x_arr, omega_t) → A·cos(kx − ωt).
    # Use phase shift π/2 so the snapshot at t=0 starts at zero-crossing:
    # cos(kx − ωt − π/2) = sin(kx − ωt).
    x = np.ascontiguousarray(np.linspace(0, 5 * LAMBDA0, 1000))
    k = OMEGA0 / C0   # wavenumber [rad/m]: parameter required by kw.plane_wave_pressure_1d
    times = [0.0, 0.25e-6, 0.5e-6, 0.75e-6]  # four quarter-period instants
    labels = [r"$t = 0$", r"$t = T/4$", r"$t = T/2$", r"$t = 3T/4$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for t, lbl, col in zip(times, labels, colors):
        # Plane-wave pressure field via kw (Kinsler et al. 2000 §5.1):
        # p(x,t) = P₀·sin(kx − ωt) = P₀·cos(kx − ωt − π/2)
        p = np.asarray(kw.plane_wave_pressure_1d(1.0, k, x, OMEGA0 * t + np.pi / 2))
        ax.plot(x * 1e3, p, color=col, label=lbl)

    ax.set_xlabel("Position $x$ (mm)")
    ax.set_ylabel("Normalised pressure $p/P_0$")
    ax.set_title("Plane-wave propagation at four time instants\n"
                 r"$p(x,t) = P_0\sin(kx - \omega t)$,  $f = 1\,\mathrm{MHz}$, $c = 1500\,\mathrm{m/s}$")
    ax.legend(ncol=2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    savefig("fig01_plane_wave_snapshots")
    plt.close(fig)


# ── Figure 02: Spherical vs cylindrical spreading ────────────────────────────
def fig02_spreading_laws() -> None:
    # Intensity derived from kw.spherical_wave_pressure (Pierce 1989 §1.6):
    #   p_sph(r) = A·cos(k·r)/r  → I ∝ p² ∝ cos²(kr)/r² → envelope 1/r²
    # For the envelope comparison, use |p|² and normalise; this correctly
    # represents the geometric 1/r² spherical spreading law from Rust physics.
    # Cylindrical spreading I ∝ 1/r is the 2-D Green's function (Pierce §1.7).
    # kw.spherical_wave_pressure gives the 3-D (spherical) result; the
    # cylindrical law is derived as I_cyl ∝ |p_sph| * sqrt(r) squared → 1/r.
    k_dc = 1e-6  # near-DC wavenumber → cos(kr) ≈ 1 → clean 1/r envelope
    r = np.ascontiguousarray(np.linspace(0.01, 0.20, 500))
    p_sph_raw = np.asarray(kw.spherical_wave_pressure(1.0, k_dc, r))
    I_spherical = p_sph_raw ** 2   # ∝ 1/r² at k→0
    I_cylindrical = np.abs(p_sph_raw)   # ∝ 1/r at k→0  (2-D source envelope)
    # normalise to 1 at r_min
    I_spherical /= I_spherical[0]
    I_cylindrical /= I_cylindrical[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(r * 100, I_spherical, label=r"Spherical: $I \propto 1/r^2$")
    ax.semilogy(r * 100, I_cylindrical, label=r"Cylindrical: $I \propto 1/r$", linestyle="--")
    ax.set_xlabel("Distance from source $r$ (cm)")
    ax.set_ylabel("Relative intensity (normalised)")
    ax.set_title("Geometric spreading laws")
    ax.legend()
    fig.tight_layout()
    savefig("fig02_spreading_laws")
    plt.close(fig)


# ── Figure 03: Normal-incidence reflection/transmission at water-tissue ───────
def fig03_reflection_transmission() -> None:
    """
    For two-medium interface: R = (Z2-Z1)/(Z2+Z1), T = 2Z2/(Z2+Z1).
    Intensity coefficients: R_I = R², T_I = 1 - R² = T·Z1/Z2.
    Sweep Z2/Z1 from 0.1 to 10.
    """
    Z1 = 1.0  # normalised
    ratio = np.linspace(0.1, 10.0, 800)
    Z2 = ratio * Z1

    # kw.reflection_pressure_coeff / transmission_pressure_coeff: scalar → vectorize.
    R_p = np.array([kw.reflection_pressure_coeff(Z1, z2) for z2 in Z2])
    T_p = np.array([kw.transmission_pressure_coeff(Z1, z2) for z2 in Z2])
    R_I = R_p**2
    T_I = 1.0 - R_I  # energy conservation

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    axes[0].plot(ratio, R_p, label=r"$R_p$ (pressure)")
    axes[0].plot(ratio, T_p, "--", label=r"$T_p$ (pressure)")
    axes[0].axvline(1.0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_xlabel(r"Impedance ratio $Z_2/Z_1$")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Pressure reflection/transmission")
    axes[0].legend()

    axes[1].plot(ratio, R_I, label=r"$R_I$ (intensity)")
    axes[1].plot(ratio, T_I, "--", label=r"$T_I$ (intensity)")
    axes[1].axvline(1.0, color="k", linewidth=0.5, linestyle=":")
    axes[1].set_xlabel(r"Impedance ratio $Z_2/Z_1$")
    axes[1].set_ylabel("Intensity coefficient")
    axes[1].set_title("Intensity reflection/transmission")
    axes[1].legend()

    fig.suptitle("Normal-incidence reflection and transmission (two-medium interface)", y=1.01)
    fig.tight_layout()
    savefig("fig03_reflection_transmission")
    plt.close(fig)


# ── Figure 04: Power-law absorption for tissue types ─────────────────────────
def fig04_power_law_absorption() -> None:
    """
    α(f) = α₀ · f^y  [dB/cm], f in MHz.
    Tissue data from Duck (1990) Table 4.1.
    Computed via kw.absorption_power_law_db_cm (Rust kernel).
    """
    f_MHz = np.linspace(0.5, 10.0, 500)
    tissues = [
        ("Water (20 °C)",  0.002, 2.0, "#1f77b4"),
        ("Blood",           0.21,  1.21, "#d62728"),
        ("Soft tissue (avg)", 0.54, 1.0, "#2ca02c"),
        ("Fat",            0.48,  1.0, "#ff7f0e"),
        ("Liver",          0.60,  1.0, "#9467bd"),
        ("Skull bone",     4.0,   1.0, "#8c564b"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, alpha0, y, col in tissues:
        alpha = np.asarray(kw.absorption_power_law_db_cm(f_MHz, alpha0, y))
        ax.semilogy(f_MHz, alpha, label=name, color=col)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(r"Absorption $\alpha$ (dB/cm)")
    ax.set_title(r"Power-law absorption: $\alpha(f) = \alpha_0 f^y$")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig04_power_law_absorption")
    plt.close(fig)


# ── Figure 05: PSTD vs FDTD phase velocity error ─────────────────────────────
def fig05_phase_velocity_error() -> None:
    """
    FDTD (second-order centred difference): k_num = (2/Δx) arcsin(Δx k / 2)
    PSTD (spectral): k_num = k (exact up to Nyquist by construction)
    Phase velocity error: ε = (c_num - c) / c = (k_num/k - 1)
    Parameterise by kΔx ∈ (0, π] (Nyquist at kΔx = π).
    """
    kdx = np.ascontiguousarray(np.linspace(0.01, np.pi - 0.01, 800))  # kΔx

    # FDTD phase error via kw.fdtd_phase_error_1d (Treeby & Cox 2010, §3.2):
    # At CFL=0.5 (representative 1-D stability margin).
    CFL_1D = 0.5
    err_fdtd = np.asarray(kw.fdtd_phase_error_1d(kdx, CFL_1D))   # relative phase error

    # PSTD phase error via kw.pstd_phase_error: zero below Nyquist by construction.
    err_pstd = np.asarray(kw.pstd_phase_error(kdx))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(kdx / np.pi, err_fdtd * 100, label="FDTD (2nd-order)")
    ax.plot(kdx / np.pi, err_pstd * 100, "--", label="PSTD (spectral)", linewidth=2)
    ax.set_xlabel(r"Normalised wavenumber $k\Delta x / \pi$")
    ax.set_ylabel("Phase velocity error (%)")
    ax.set_title("Numerical dispersion: PSTD vs second-order FDTD")
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig05_phase_velocity_error")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 8 figures (Acoustic Propagation)...")
    fig01_plane_wave_snapshots()
    fig02_spreading_laws()
    fig03_reflection_transmission()
    fig04_power_law_absorption()
    fig05_phase_velocity_error()
    print(f"Done. Output: docs/book/figures/ch08/")
