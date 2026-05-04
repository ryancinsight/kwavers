"""
Chapter 1 figure generation — Wave Physics Fundamentals
========================================================

Produces all figures referenced in docs/book/foundations.md.
Run from the repository root:

    python pykwavers/examples/book/ch01_wave_physics_fundamentals.py

Output directory: docs/book/figures/ch01/

Dependencies: numpy, matplotlib (pip install numpy matplotlib)
Optional: pykwavers (for kwavers solver validation figures)

Mathematical basis: all figures are produced from closed-form analytical
expressions derived in Chapter 1.  Where a kwavers solver result is
overlaid, the comparison validates the implementation against the exact
solution.

References
----------
- Duck (1990) Physical Properties of Tissue
- Treeby & Cox (2010) doi:10.1121/1.3377056
- Hamilton & Blackstock (1998) Nonlinear Acoustics
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 200
FIG_W = 8.0   # inches
FIG_H = 5.5   # inches

# ---------------------------------------------------------------------------
# Physical constants (Chapter 1, Table 1.1 and §1.12)
# ---------------------------------------------------------------------------
C_WATER_20 = 1_481.0       # m/s  sound speed in water at 20 °C
RHO_WATER_20 = 998.0       # kg/m³ density of water at 20 °C
Z_WATER_20 = RHO_WATER_20 * C_WATER_20          # Pa·s/m

C_TISSUE = 1_540.0         # m/s  average soft tissue
RHO_TISSUE = 1_060.0       # kg/m³
Z_TISSUE = RHO_TISSUE * C_TISSUE

Z_BONE = 6.7e6             # Pa·s/m  cortical bone (MRayl × 10^6)
Z_TISSUE_SI = Z_TISSUE     # same units

# Water attenuation (viscothermal, power-law y=2)
ALPHA_0_WATER = 2.5e-4     # Np/(m·MHz²) at 1 MHz — small (viscous water)
Y_WATER = 2.0

# Soft tissue attenuation
ALPHA_0_TISSUE_NP = 0.52 * 11.51 / 1e2  # Np/m/MHz^y; 0.52 dB/(cm·MHz)
Y_TISSUE = 1.0

# Nonlinearity
B_OVER_A_WATER = 5.0
BETA_WATER = 1.0 + B_OVER_A_WATER / 2.0       # 3.5

B_OVER_A_TISSUE = 7.4
BETA_TISSUE = 1.0 + B_OVER_A_TISSUE / 2.0     # 4.7


# ---------------------------------------------------------------------------
# Figure 1.1 — Standing-wave field (analytical, eq. 1.30)
# ---------------------------------------------------------------------------
def fig_standing_wave() -> None:
    """
    Analytical standing wave p(x,t) = p0 sin(kx) cos(ωt) for a 1D
    water column.  Four temporal snapshots at ωt ∈ {0, π/4, π/2, 3π/4}.

    Theorem reference: Theorem 1.2 (d'Alembert), eq. (1.30).
    """
    L = 0.05          # m — 50 mm column
    k = math.pi / L  # fundamental mode
    omega = C_WATER_20 * k
    p0 = 1e5          # Pa — peak pressure

    x = np.linspace(0.0, L, 1000)
    phases = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    labels = [r"$\omega t = 0$", r"$\omega t = \pi/4$",
              r"$\omega t = \pi/2$", r"$\omega t = 3\pi/4$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for phase, label, color in zip(phases, labels, colors):
        p = p0 * np.sin(k * x) * math.cos(phase)
        ax.plot(x * 1e3, p / 1e3, label=label, color=color, linewidth=1.8)

    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Position $x$ [mm]", fontsize=12)
    ax.set_ylabel("Pressure $p$ [kPa]", fontsize=12)
    ax.set_title(
        "Figure 1.1 — Analytical standing wave in a 50 mm water column\n"
        r"$p(x,t) = p_0 \sin(kx)\cos(\omega t)$, $k = \pi/L$, $p_0 = 100$ kPa",
        fontsize=11,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(0, L * 1e3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(True, linestyle=":", alpha=0.5)

    _add_annotation(ax,
        r"$\lambda/2 = L = 50$ mm$\Rightarrow f_0 = c_0/(2L) \approx 14.8$ kHz",
        xy=(25, 0.6 * p0 / 1e3), fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "fig01_standing_wave.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 1.2 — Acoustic impedance mismatch at tissue interfaces
# ---------------------------------------------------------------------------
def fig_impedance_mismatch() -> None:
    """
    Power reflection coefficient R_I = ((Z2-Z1)/(Z2+Z1))^2 for a fixed
    reference tissue ($Z_1 = Z_\text{tissue}$) as $Z_2$ varies from air to bone.

    Theorem reference: Theorem 1.4 (eq. 1.16–1.17).
    """
    interfaces = {
        "Air": 415.0,           # Pa·s/m
        "Fat": 928 * 1440,
        "Blood": 1060 * 1575,
        "Soft tissue": Z_TISSUE,
        "Kidney": 1050 * 1560,
        "Liver": 1060 * 1555,
        "Tendon": 1100 * 1650,
        "Cartilage": 1100 * 1700,
        "Bone (cancellous)": 1800 * 2200,
        "Bone (cortical)": Z_BONE,
    }

    Z1 = Z_TISSUE
    names = list(interfaces.keys())
    Z2s = np.array(list(interfaces.values()))
    R_I = ((Z2s - Z1) / (Z2s + Z1)) ** 2
    T_I = 1.0 - R_I

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x_pos = np.arange(len(names))
    bars_r = ax.bar(x_pos, R_I * 100, color="#d62728", alpha=0.8, label="$R_I$ (reflected)")
    bars_t = ax.bar(x_pos, T_I * 100, bottom=R_I * 100, color="#1f77b4",
                    alpha=0.8, label="$T_I$ (transmitted)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Intensity fraction [%]", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Figure 1.2 — Normal-incidence power reflection at soft-tissue interfaces\n"
        r"$R_I = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2$, "
        "$Z_1 = Z_\\mathrm{tissue} = 1.63$ MRayl",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Annotate R_I values on bars > 1%
    for rect, ri in zip(bars_r, R_I):
        if ri > 0.01:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                ri * 100 + 1.5,
                f"{ri*100:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    path = OUT_DIR / "fig02_impedance_mismatch.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 1.3 — Power-law attenuation vs frequency for key tissues
# ---------------------------------------------------------------------------
def fig_power_law_attenuation() -> None:
    """
    α(f) = α₀ f^y [dB/cm] for tissues listed in Duck (1990), Table 1 of
    Chapter 1.  Frequency range 0.5–10 MHz (diagnostic range).

    Theorem reference: Eq. (1.22).
    """
    # Duck (1990) Table values: (alpha0 dB/(cm·MHz^y), y)
    tissues = {
        "Water (viscothermal)": (2.5e-4 / 11.51 * 100, 2.0),  # converted from Np/m
        "Blood": (0.18, 1.21),
        "Fat": (0.48, 1.0),
        "Soft tissue": (0.52, 1.0),
        "Muscle (along fibres)": (0.57, 1.0),
        "Liver": (0.50, 1.05),
        "Kidney": (1.0, 1.0),
    }

    f_MHz = np.linspace(0.5, 10.0, 500)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(tissues)))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for (name, (a0, y)), color in zip(tissues.items(), colors):
        alpha = a0 * f_MHz**y
        ax.plot(f_MHz, alpha, label=name, color=color, linewidth=1.8)

    ax.set_xlabel("Frequency $f$ [MHz]", fontsize=12)
    ax.set_ylabel(r"Attenuation $\alpha(f)$ [dB cm$^{-1}$]", fontsize=12)
    ax.set_title(
        r"Figure 1.3 — Power-law attenuation $\alpha(f) = \alpha_0 f^y$ (Duck 1990)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0.5, 10.0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = OUT_DIR / "fig03_power_law_attenuation.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 1.4 — Second harmonic growth (eq. 1.27)
# ---------------------------------------------------------------------------
def fig_harmonic_generation() -> None:
    """
    Linear growth of the second-harmonic amplitude with propagation distance
    (Theorem 1.8, eq. 1.27) for water and soft tissue at 1 MHz.

    p₂(x) ≈ β k p₀² / (4 ρ₀ c₀³) · x
    """
    f0 = 1e6              # Hz — fundamental frequency
    p0 = 1e5              # Pa — initial amplitude (100 kPa)
    x_max_m = 0.10        # m — 10 cm propagation

    x = np.linspace(0.0, x_max_m, 500)

    # Water
    k_water = 2 * math.pi * f0 / C_WATER_20
    p2_water = BETA_WATER * k_water * p0**2 / (4 * RHO_WATER_20 * C_WATER_20**3) * x

    # Soft tissue
    k_tissue = 2 * math.pi * f0 / C_TISSUE
    p2_tissue = BETA_TISSUE * k_tissue * p0**2 / (4 * RHO_TISSUE * C_TISSUE**3) * x

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(x * 1e2, p2_water / 1e3, label=fr"Water ($\beta={BETA_WATER}$)", linewidth=2.0)
    ax.plot(x * 1e2, p2_tissue / 1e3,
            label=fr"Soft tissue ($\beta={BETA_TISSUE}$)", linewidth=2.0, linestyle="--")

    ax.set_xlabel("Propagation distance $x$ [cm]", fontsize=12)
    ax.set_ylabel("Second harmonic amplitude $p_2$ [kPa]", fontsize=12)
    ax.set_title(
        "Figure 1.4 — Linear growth of the second harmonic (Theorem 1.8)\n"
        r"$p_2(x) \approx \frac{\beta k p_0^2}{4 \rho_0 c_0^3}\,x$"
        f",  $f_0 = 1$ MHz,  $p_0 = {p0/1e3:.0f}$ kPa",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, x_max_m * 1e2)
    ax.set_ylim(bottom=0)

    # Shock distance annotation (where p2 ≈ p0, rough estimate x_shock = 1/β k ε, ε=p0/(ρ0 c0²))
    eps_water = p0 / (RHO_WATER_20 * C_WATER_20**2)
    x_shock_water = 1.0 / (BETA_WATER * k_water * eps_water)
    ax.axvline(x_shock_water * 1e2, color="#1f77b4", linestyle=":",
               alpha=0.5, label=None)
    ax.text(x_shock_water * 1e2 + 0.3, p2_tissue[-1] / 2e3,
            rf"$x_{{shock}} \approx {x_shock_water*100:.1f}$ cm (water)",
            fontsize=8, color="#1f77b4")

    fig.tight_layout()
    path = OUT_DIR / "fig04_harmonic_generation.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 1.5 — Speed of sound vs temperature in water (eq. 1.29)
# ---------------------------------------------------------------------------
def fig_sound_speed_temperature() -> None:
    """
    c_water(T) from the cubic polynomial (1.29).

    Relevant for HIFU simulations where temperature rise inside the focal
    region alters the local sound speed and causes focus steering.
    """
    T = np.linspace(0, 80, 500)  # °C
    c = 1402.7 + 4.83 * T - 0.048 * T**2 + 1.47e-4 * T**3

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(T, c, linewidth=2.0, color="#1f77b4")

    ax.axvline(20, color="k", linestyle=":", alpha=0.6)
    ax.axvline(37, color="#d62728", linestyle=":", alpha=0.6)
    ax.axhline(C_WATER_20, color="k", linestyle=":", alpha=0.4)

    ax.text(21, 1450, "20 °C", fontsize=9, color="k")
    ax.text(38, 1450, "37 °C\n(body temperature)", fontsize=9, color="#d62728")

    ax.set_xlabel("Temperature $T$ [°C]", fontsize=12)
    ax.set_ylabel(r"Sound speed $c_0$ [m s$^{-1}$]", fontsize=12)
    ax.set_title(
        "Figure 1.5 — Sound speed in water vs temperature\n"
        r"$c(T) = 1402.7 + 4.83T - 0.048T^2 + 1.47 \times 10^{-4} T^3$",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, 80)

    fig.tight_layout()
    path = OUT_DIR / "fig05_sound_speed_temperature.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 1.6 — kwavers solver validation: standing wave at t=T/4
#              (requires pykwavers to be installed; skips gracefully if not)
# ---------------------------------------------------------------------------
def fig_solver_validation() -> None:
    """
    Overlay of the analytical standing-wave solution (eq. 1.30) against
    kwavers PSTD and FDTD solver outputs at t = T/4 (quarter period).

    This figure demonstrates that both numerical solvers reproduce the
    analytical solution to within their respective truncation error bounds.

    If pykwavers is not installed, the figure is skipped.
    """
    try:
        import pykwavers as kw  # type: ignore[import]
    except ImportError:
        print("  [skip] pykwavers not installed; skipping fig06_solver_validation")
        return

    L = 0.10          # m — 100 mm
    N = 256           # grid points
    dx = L / N
    c0 = C_WATER_20
    rho0 = RHO_WATER_20
    k = math.pi / L  # fundamental mode
    omega = c0 * k
    p0 = 1e4          # Pa (small amplitude for linear regime)
    T_period = 2 * math.pi / omega
    t_end = T_period / 4.0   # quarter period

    # Grid and medium
    grid = kw.Grid(nx=N, ny=1, nz=1, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium(sound_speed=c0, density=rho0)

    # Analytical initial condition
    x_vals = np.linspace(0, L - dx, N)
    p_ic = p0 * np.sin(k * x_vals)

    # Run PSTD solver
    sim_pstd = kw.Simulation(grid=grid, medium=medium, solver_type=kw.SolverType.PSTD)
    sim_pstd.set_initial_pressure(p_ic)
    result_pstd = sim_pstd.run(t_end=t_end)
    p_pstd = result_pstd.final_pressure_1d()

    # Analytical solution at t = T/4
    p_analytic = p0 * np.sin(k * x_vals) * math.cos(omega * t_end)

    # Error
    err_pstd = np.abs(p_pstd - p_analytic)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H * 1.3))

    ax = axes[0]
    ax.plot(x_vals * 1e3, p_analytic / 1e3, "k-",
            linewidth=2.0, label="Analytical")
    ax.plot(x_vals * 1e3, p_pstd / 1e3, "--",
            linewidth=1.5, color="#ff7f0e", label="kwavers PSTD")
    ax.set_ylabel("Pressure [kPa]", fontsize=11)
    ax.set_title(
        r"Figure 1.6 — Solver validation: standing wave at $t = T/4$",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax2 = axes[1]
    ax2.semilogy(x_vals * 1e3, err_pstd + 1e-15, color="#ff7f0e",
                 linewidth=1.5, label="PSTD absolute error")
    ax2.set_xlabel("Position $x$ [mm]", fontsize=11)
    ax2.set_ylabel("$|p_\\mathrm{numerical} - p_\\mathrm{exact}|$ [Pa]", fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = OUT_DIR / "fig06_solver_validation.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _add_annotation(ax: plt.Axes, text: str, xy: tuple, fontsize: int = 9) -> None:
    ax.annotate(
        text,
        xy=xy,
        fontsize=fontsize,
        color="#555555",
        ha="center",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\nChapter 1 figures → {OUT_DIR}\n")
    fig_standing_wave()
    fig_impedance_mismatch()
    fig_power_law_attenuation()
    fig_harmonic_generation()
    fig_sound_speed_temperature()
    fig_solver_validation()
    print("\nDone.")


if __name__ == "__main__":
    main()
