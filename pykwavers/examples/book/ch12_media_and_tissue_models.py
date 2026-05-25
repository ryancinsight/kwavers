"""
Chapter 12 figure generation — Media and Tissue Models
=======================================================

Produces publication-quality figures for docs/book/media_and_tissue_models.md.
Data from Duck (1990), ICRU Report 61 (1998), Mast (2000).

Output directory: docs/book/figures/ch12/

Figures produced
----------------
fig01  Speed of sound vs temperature: water and soft tissue
fig02  Acoustic impedance Z = ρc for major tissue types (horizontal bar)
fig03  Nonlinearity parameter B/A for tissue types
fig04  Fractional-Laplacian absorption: α(ω) vs ω for y = 1.0, 1.5, 2.0
fig05  Pennes bioheat: temperature rise vs depth (steady-state, 1D)

References
----------
Duck (1990) Physical Properties of Tissue. Academic Press.
Mast (2000) J. Acoust. Soc. Am. 107(6):3384
ICRU Report 61 (1998) Tissue Substitutes, Phantoms and Computational Models
Treeby & Cox (2010) doi:10.1121/1.3377056
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch12")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch12/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})


# ── Figure 01: Speed of sound vs temperature ─────────────────────────────────
def fig01_sound_speed_temperature() -> None:
    """
    Water: Del Grosso & Mader (1972) polynomial via kw.water_sound_speed_temperature.
    Soft tissue: Bamber & Hill (1979) linear approx c = 1540 + 1.8·(T−37) [no kw binding].
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig01 (water sound speed)")
    T = np.linspace(0, 60, 300)
    c_water = np.asarray(kw.water_sound_speed_temperature(T))
    # Bamber & Hill (1979) linear approximation — no temperature-dependent tissue binding.
    c_tissue = 1540 + 1.8 * (T - 37)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T, c_water, label="Water (Del Grosso & Mader 1972)")
    ax.plot(T, c_tissue, "--", label="Soft tissue (Bamber & Hill 1979)")
    ax.axvline(37, color="k", linewidth=0.5, linestyle=":", label="Body temperature 37 °C")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Speed of sound (m/s)")
    ax.set_title("Speed of sound vs temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig01_sound_speed_temperature")
    plt.close(fig)


# ── Figure 02: Acoustic impedance by tissue ───────────────────────────────────
def fig02_impedance_bar() -> None:
    """
    Z = ρ·c   [MRayl = 10⁶ kg/(m²·s)]
    Computed via kw.tissue_properties(name) → (c, rho, ...) → Z = c*rho/1e6.
    Air and Cortical bone have no kw binding; Duck (1990) reference values used.
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig02 (tissue impedance)")

    def _z_mrayl(tissue_key: str) -> float:
        """Acoustic impedance [MRayl] from kw.tissue_properties."""
        c, rho, *_ = kw.tissue_properties(tissue_key)
        return c * rho / 1e6

    tissues = [
        ("Air",           0.0004),           # Duck (1990) — no kw binding for gas phase.
        ("Fat",           _z_mrayl("fat")),
        ("Water (37°C)",  _z_mrayl("water")),
        ("Blood",         _z_mrayl("blood")),
        ("Brain",         _z_mrayl("brain")),
        ("Liver",         _z_mrayl("liver")),
        ("Kidney",        _z_mrayl("kidney")),
        ("Muscle",        _z_mrayl("muscle")),
        ("Cartilage",     _z_mrayl("cartilage")),
        ("Cortical bone", 7.38),             # Duck (1990) — use "skull" bone reference.
    ]
    names = [t for t, _ in tissues]
    Z = np.array([z for _, z in tissues])

    colors = plt.cm.viridis(Z / Z.max())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, Z, color=colors)
    ax.set_xlabel("Acoustic impedance Z (MRayl)")
    ax.set_title("Acoustic impedance $Z = \\rho c$ by tissue type")
    ax.axvline(1.52, color="blue", linestyle="--", linewidth=1, label="Water")
    ax.legend()
    for bar, val in zip(bars, Z):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    savefig("fig02_impedance_bar")
    plt.close(fig)


# ── Figure 03: B/A nonlinearity parameter ────────────────────────────────────
def fig03_ba_parameter() -> None:
    """
    B/A = 2ρc ∂c/∂p|_s  — acoustic nonlinearity.
    β = 1 + B/(2A).
    Computed via kw.ba_parameter(tissue_key) (Hamilton & Blackstock 1998, Rust kernel).
    Water is not temperature-dependent in the binding (single canonical value).
    Breast tissue uses "fat" binding (high fat content).
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig03 (B/A parameter)")
    tissues = [
        ("Water (20°C)", kw.ba_parameter("water")),
        ("Water (37°C)", kw.ba_parameter("water")),
        ("Blood",        kw.ba_parameter("blood")),
        ("Fat",          kw.ba_parameter("fat")),
        ("Liver",        kw.ba_parameter("liver")),
        ("Muscle",       kw.ba_parameter("muscle")),
        ("Brain",        kw.ba_parameter("brain")),
        ("Kidney",       kw.ba_parameter("kidney")),
        ("Breast (fat)", kw.ba_parameter("fat")),   # high fat content → use fat B/A
    ]
    names = [t for t, _ in tissues]
    BA = np.array([b for _, b in tissues])
    beta = 1 + BA / 2

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, BA, label="B/A", color="#1f77b4", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(x, beta, "o-", color="#d62728", label=r"$\beta = 1 + B/2A$")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("B/A (dimensionless)")
    ax2.set_ylabel(r"Nonlinearity coefficient $\beta$", color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    ax.set_title("Acoustic nonlinearity B/A by tissue type")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    fig.tight_layout()
    savefig("fig03_ba_parameter")
    plt.close(fig)


# ── Figure 04: Fractional-Laplacian absorption ───────────────────────────────
def fig04_fractional_absorption() -> None:
    """
    Power-law absorption: α(f) = α₀ f^y  [Np/m, f in Hz]
    Fractional Laplacian (Treeby & Cox 2010): y ∈ {1.0, 1.5, 2.0}.
    Computed via kw.power_law_attenuation_np_m(f_hz, alpha0, y), normalised at 1 MHz.
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig04 (power-law absorption)")
    f_MHz = np.linspace(0.5, 10.0, 400)
    f_hz = f_MHz * 1e6
    alpha0 = 1.0  # Np/m at 1 Hz — normalised for shape illustration
    f_ref_hz = 1.0e6  # normalise at 1 MHz

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for y, col, lbl in [(1.0, "#1f77b4", r"$y=1.0$ (tissue-like)"),
                        (1.5, "#ff7f0e", r"$y=1.5$"),
                        (2.0, "#2ca02c", r"$y=2.0$ (thermoviscous)")]:
        alpha = np.asarray(kw.power_law_attenuation_np_m(f_hz, alpha0, y))
        # Normalise at 1 MHz so curves are dimensionless and comparable
        alpha_ref = float(np.asarray(kw.power_law_attenuation_np_m(np.array([f_ref_hz]), alpha0, y))[0])
        alpha_norm = alpha / alpha_ref
        ax.loglog(f_MHz, alpha_norm, color=col, label=lbl)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(r"Normalised absorption $\alpha/\alpha_0(1\,\mathrm{MHz})$")
    ax.set_title(r"Power-law absorption: $\alpha(\omega) = \alpha_0 |\omega|^y$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig04_fractional_absorption")
    plt.close(fig)


# ── Figure 05: Pennes bioheat steady-state 1D ────────────────────────────────
def fig05_bioheat() -> None:
    """
    Pennes bioheat equation (steady-state, 1D, uniform heating Q):
      k ∂²T/∂x² - W_b ρ_b c_b (T - T_b) + Q = 0
    Analytical solution with Dirichlet BC T(0)=T(L)=T_b:
      T(x) = T_b + Q/(W_b ρ_b c_b) · (1 - cosh(α(x-L/2))/cosh(αL/2))
    where α = √(W_b ρ_b c_b / k).
    """
    L = 0.04    # 40 mm tissue thickness
    k_t = 0.5   # W/(m·K) thermal conductivity
    Wb = 0.005  # 1/s  blood perfusion rate (kg_blood/(kg_tissue·s))
    rho_b = 1060.0  # kg/m³ blood density
    c_b = 3617.0    # J/(kg·K) blood specific heat
    T_b = 37.0  # °C blood/body temperature
    Q_vals = [(1e4, "#1f77b4", r"$Q=10\,\mathrm{kW/m^3}$"),
              (5e4, "#ff7f0e", r"$Q=50\,\mathrm{kW/m^3}$"),
              (2e5, "#2ca02c", r"$Q=200\,\mathrm{kW/m^3}$")]

    x = np.linspace(0, L, 400)
    alpha_perf = np.sqrt(Wb * rho_b * c_b / k_t)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for Q, col, lbl in Q_vals:
        dT_max = Q / (Wb * rho_b * c_b)
        T_x = T_b + dT_max * (1 - np.cosh(alpha_perf * (x - L / 2)) / np.cosh(alpha_perf * L / 2))
        ax.plot(x * 1e3, T_x, color=col, label=lbl)

    ax.axhline(T_b, color="k", linewidth=0.5, linestyle="--", label=r"$T_b = 37\,°C$")
    ax.set_xlabel("Depth $x$ (mm)")
    ax.set_ylabel("Temperature $T$ (°C)")
    ax.set_title("Pennes bioheat equation: 1D steady-state\n"
                 r"$k\nabla^2T - W_b\rho_bc_b(T-T_b) + Q = 0$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig05_bioheat")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 12 figures (Media and Tissue Models)...")
    fig01_sound_speed_temperature()
    fig02_impedance_bar()
    fig03_ba_parameter()
    fig04_fractional_absorption()
    fig05_bioheat()
    print("Done. Output: docs/book/figures/ch12/")
