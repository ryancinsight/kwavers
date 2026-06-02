"""
Chapter 10 figure generation — Elastography
============================================

Produces publication-quality figures for docs/book/elastography.md.
All expressions are closed-form analytical solutions from linear elasticity
and viscoelastic theory.

Output directory: docs/book/figures/ch10/

Figures produced
----------------
fig01  Shear wave speed vs shear modulus (normal and pathological tissue)
fig02  P-wave vs S-wave velocity ratio as function of Poisson's ratio
fig03  Voigt model: storage G'(ω) and loss G''(ω) moduli vs frequency
fig04  Wave speed dispersion: shear wave group vs phase velocity (Voigt)
fig05  MRE harmonic displacement field (cylindrical scatterer, analytical)

References
----------
Sinkus et al. (2005) doi:10.1016/j.mri.2005.03.017
Muthupillai et al. (1995) doi:10.1126/science.7716562
Manduca et al. (2001) doi:10.1016/S1361-8415(00)00039-0
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch10")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch10/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

RHO = 1060.0   # kg/m³ average soft tissue density


# ── Figure 01: Shear wave speed vs shear modulus ─────────────────────────────
def fig01_shear_wave_speed() -> None:
    """
    c_s = √(G / ρ)  — shear wave phase velocity.
    Computed via kw.shear_wave_speed(g_pa, rho_kg_m3) (Rust kernel).
    Representative tissues from Sinkus 2005 and Deffieux 2011.
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig01 (shear wave speed)")
    tissues = [
        ("Brain (GM)",    2.0e3),
        ("Liver (normal)", 5.0e3),
        ("Liver (fibrosis)", 20e3),
        ("Breast (fat)",    2.5e3),
        ("Breast (tumour)", 50e3),
        ("Muscle (along)", 40e3),
        ("Thyroid (normal)", 8e3),
        ("Prostate (normal)", 10e3),
    ]
    G_Pa = np.array([g for _, g in tissues])
    c_s = np.array([kw.shear_wave_speed(G, RHO) for G in G_Pa])

    G_range = np.logspace(2, 6, 300)  # 100 Pa → 1 MPa
    c_range = np.array([kw.shear_wave_speed(G, RHO) for G in G_range])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(G_range, c_range, "k-", linewidth=1.5, label=r"$c_s = \sqrt{G/\rho}$")
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(tissues)))
    for (name, G), col in zip(tissues, colors):
        cs = kw.shear_wave_speed(G, RHO)
        ax.scatter(G, cs, s=80, color=col, zorder=5)
        ax.annotate(name, (G, cs), textcoords="offset points", xytext=(5, 3), fontsize=8, color=col)

    ax.set_xlabel(r"Shear modulus $G$ (Pa)")
    ax.set_ylabel(r"Shear wave speed $c_s$ (m/s)")
    ax.set_title(r"Shear wave speed: $c_s = \sqrt{G/\rho}$, $\rho = 1060\,\mathrm{kg/m^3}$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig01_shear_wave_speed")
    plt.close(fig)


# ── Figure 02: P-wave vs S-wave velocity ratio vs Poisson's ratio ─────────────
def fig02_wave_velocity_ratio() -> None:
    """
    c_p / c_s = √((1-ν)/((1-2ν)·(1+ν)/2)) = √(2(1-ν)/(1-2ν))
    For incompressible tissue ν→0.5: c_p/c_s → ∞.
    """
    nu = np.linspace(0.0, 0.499, 500)
    ratio = np.sqrt(2 * (1 - nu) / (1 - 2 * nu))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(nu, ratio, color="#1f77b4")
    ax.axvline(0.499, color="gray", linestyle=":", linewidth=1)
    ax.text(0.48, 30, r"$\nu \to 0.5$ (tissue)", fontsize=9, ha="right", color="gray")
    ax.axhline(1.0, color="k", linewidth=0.5)
    ax.set_xlabel(r"Poisson's ratio $\nu$")
    ax.set_ylabel(r"Velocity ratio $c_p / c_s$")
    ax.set_title(r"P- to S-wave velocity ratio: $c_p/c_s = \sqrt{2(1-\nu)/(1-2\nu)}$")
    ax.set_ylim(1, 200)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig02_wave_velocity_ratio")
    plt.close(fig)


# ── Figure 03: Voigt model storage and loss moduli ───────────────────────────
def fig03_voigt_viscoelastic() -> None:
    """
    Voigt model: G*(ω) = G_e + iωη.
    Storage modulus: G'(ω) = Re G* = G_e   (frequency-independent).
    Loss modulus:   G''(ω) = Im G* = ω η.
    Loss angle:     tan δ = G''/G' = ωη/G_e.
    Computed via kw.voigt_complex_modulus(omega_arr, mu_pa, eta_pa_s) → (real_arr, imag_arr).
    Representative values: G_e = 2 kPa, η = 1 Pa·s.
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig03 (Voigt viscoelastic model)")
    f_Hz = np.logspace(0, 3, 500)   # 1 Hz → 1 kHz
    omega = 2 * np.pi * f_Hz

    G_e = 2e3    # Pa  elastic modulus
    eta = 1.0    # Pa·s viscosity

    G_prime_arr, G_dprime_arr = kw.voigt_complex_modulus(omega, G_e, eta)
    G_prime = np.asarray(G_prime_arr)
    G_dprime = np.asarray(G_dprime_arr)
    tan_delta = G_dprime / G_prime

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(f_Hz, G_prime, label=r"$G'(\omega)$ — storage")
    ax1.loglog(f_Hz, G_dprime, "--", label=r"$G''(\omega)$ — loss")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Modulus (Pa)")
    ax1.set_title(r"Voigt model: $G^* = G_e + i\omega\eta$"
                  f"\n$G_e={G_e/1e3:.0f}$ kPa, $\\eta={eta}$ Pa·s")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(f_Hz, np.degrees(np.arctan(tan_delta)), color="#d62728")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(r"Loss angle $\delta$ (°)")
    ax2.set_title(r"Loss angle $\delta = \arctan(G''/G')$")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    savefig("fig03_voigt_viscoelastic")
    plt.close(fig)


# ── Figure 04: Shear wave dispersion (Voigt model) ───────────────────────────
def fig04_shear_dispersion() -> None:
    """
    Complex shear modulus G*(ω) = G_e + iωη.
    Computed via kw.voigt_complex_modulus(omega_arr, G_e, eta) → (real_arr, imag_arr).
    Complex wavenumber: k_s(ω) = ω √(ρ / G*(ω)).
    Phase velocity: c_ph = ω / Re(k_s).
    Elastic limit via kw.shear_wave_speed(G_e, RHO).
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig04 (shear wave dispersion)")
    f_Hz = np.logspace(0, 3, 500)
    omega = 2 * np.pi * f_Hz

    G_e = 2e3
    eta_values = [(0.1, r"$\eta=0.1$ Pa·s"), (1.0, r"$\eta=1$ Pa·s"), (5.0, r"$\eta=5$ Pa·s")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for eta, lbl in eta_values:
        G_re, G_im = kw.voigt_complex_modulus(omega, G_e, eta)
        G_star = np.asarray(G_re) + 1j * np.asarray(G_im)
        k_s = omega * np.sqrt(RHO / G_star)
        c_ph = omega / np.real(k_s)
        ax.semilogx(f_Hz, c_ph, label=lbl)

    # Elastic limit: c_s = sqrt(G_e / rho) via Rust kernel
    c_elastic = kw.shear_wave_speed(G_e, RHO)
    ax.axhline(c_elastic, color="k", linestyle=":", linewidth=1,
               label=rf"Elastic limit $c_s={c_elastic:.2f}$ m/s")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Phase velocity $c_\mathrm{ph}$ (m/s)")
    ax.set_title("Voigt shear wave dispersion\n"
                 r"$k_s(\omega) = \omega\sqrt{\rho/(G_e + i\omega\eta)}$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig04_shear_dispersion")
    plt.close(fig)


# ── Figure 05: MRE harmonic displacement field ────────────────────────────────
def fig05_mre_displacement() -> None:
    """
    Displacement field of a shear wave propagating in the x-direction
    with a cylindrical stiff inclusion (radius r_inc, G_inc > G_bg).
    Outside: plane shear wave u_y = A sin(k_s x - ωt).
    Inside scatterer: standing wave with modified wavenumber (Born approx).
    Wavenumbers: k = ω / c_s, c_s via kw.shear_wave_speed(G, rho).
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig05 (MRE displacement field)")
    nx, ny = 300, 300
    x = np.linspace(-0.05, 0.05, nx)   # ±50 mm
    y = np.linspace(-0.05, 0.05, ny)
    X, Y = np.meshgrid(x, y)

    G_bg = 2e3    # Pa background shear modulus
    G_inc = 20e3  # Pa inclusion shear modulus (tumour-like)
    r_inc = 0.01  # 10 mm radius
    rho = RHO
    f = 100.0     # 100 Hz MRE frequency
    omega = 2 * np.pi * f

    # k = ω / c_s, c_s from Rust kernel so wavenumbers are consistent with fig01/04
    c_bg = kw.shear_wave_speed(G_bg, rho)
    c_inc = kw.shear_wave_speed(G_inc, rho)
    k_bg = omega / c_bg
    k_inc = omega / c_inc

    # Incident shear wave propagating in +x
    u_incident = np.sin(k_bg * X)

    # Inside cylinder: faster phase (higher G → higher c → lower k)
    r = np.sqrt(X**2 + Y**2)
    mask_inc = r < r_inc

    u_field = u_incident.copy()
    u_field[mask_inc] = np.sin(k_inc * X[mask_inc])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].pcolormesh(x * 1e3, y * 1e3, u_field, cmap="RdBu_r", shading="auto")
    circle = plt.Circle((0, 0), r_inc * 1e3, fill=False, color="k", linewidth=1.5, linestyle="--")
    axes[0].add_patch(circle)
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title(r"MRE displacement $u_y(x,y)$" "\n"
                       rf"$G_\mathrm{{bg}}={G_bg/1e3:.0f}$ kPa, $G_\mathrm{{inc}}={G_inc/1e3:.0f}$ kPa, $f={f:.0f}$ Hz")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], label="Normalised displacement")

    # Local wavelength map
    lam_field = np.full_like(r, 2 * np.pi / k_bg)
    lam_field[mask_inc] = 2 * np.pi / k_inc
    im1 = axes[1].pcolormesh(x * 1e3, y * 1e3, lam_field * 1e3, cmap="viridis", shading="auto")
    circle2 = plt.Circle((0, 0), r_inc * 1e3, fill=False, color="w", linewidth=1.5, linestyle="--")
    axes[1].add_patch(circle2)
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].set_title("Local wavelength map\n(shorter λ → higher G)")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], label="Wavelength (mm)")

    fig.tight_layout()
    savefig("fig05_mre_displacement")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 10 figures (Elastography)...")
    fig01_shear_wave_speed()
    fig02_wave_velocity_ratio()
    fig03_voigt_viscoelastic()
    fig04_shear_dispersion()
    fig05_mre_displacement()
    print("Done. Output: docs/book/figures/ch10/")
