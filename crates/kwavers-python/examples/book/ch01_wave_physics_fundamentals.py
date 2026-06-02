"""
Chapter 1 figure generation — Wave Physics Fundamentals
========================================================

Produces all figures referenced in docs/book/foundations.md.
Run from the repository root:

    python pykwavers/examples/book/ch01_wave_physics_fundamentals.py

Output directory: docs/book/figures/ch01/

All physics computed by kwavers (Rust); this file contains only matplotlib
rendering.  Requires pykwavers to be installed.

References
----------
- Duck (1990) Physical Properties of Tissue
- Treeby & Cox (2010) doi:10.1121/1.3377056
- Hamilton & Blackstock (1998) Nonlinear Acoustics
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pykwavers as kw

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 200
FIG_W = 8.0
FIG_H = 5.5


def fig_standing_wave() -> None:
    L = 0.05
    k = math.pi / L
    p0 = 1e5
    x = np.linspace(0.0, L, 1000)
    phases = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    labels = [r"$\omega t = 0$", r"$\omega t = \pi/4$",
              r"$\omega t = \pi/2$", r"$\omega t = 3\pi/4$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for phase, label, color in zip(phases, labels, colors):
        p = kw.standing_wave_1d(p0, k, x, phase)
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
    fig.tight_layout()
    path = OUT_DIR / "fig01_standing_wave.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_impedance_mismatch() -> None:
    interfaces = {
        "Air": 415.0,
        "Fat": 928 * 1440,
        "Blood": 1060 * 1575,
        "Soft tissue": 1060 * 1540,
        "Kidney": 1050 * 1560,
        "Liver": 1060 * 1555,
        "Tendon": 1100 * 1650,
        "Cartilage": 1100 * 1700,
        "Bone (cancellous)": 1800 * 2200,
        "Bone (cortical)": 6.7e6,
    }
    Z1 = 1060.0 * 1540.0
    names = list(interfaces.keys())
    Z2s = np.array(list(interfaces.values()))

    R_I = np.array([kw.reflection_pressure_coeff(Z1, z2) ** 2 for z2 in Z2s])
    T_I = 1.0 - R_I

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x_pos = np.arange(len(names))
    bars_r = ax.bar(x_pos, R_I * 100, color="#d62728", alpha=0.8, label="$R_I$ (reflected)")
    ax.bar(x_pos, T_I * 100, bottom=R_I * 100, color="#1f77b4",
           alpha=0.8, label="$T_I$ (transmitted)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Intensity fraction [%]", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Figure 1.2 — Normal-incidence power reflection at soft-tissue interfaces\n"
        r"$R_I = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2$",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    for rect, ri in zip(bars_r, R_I):
        if ri > 0.01:
            ax.text(rect.get_x() + rect.get_width() / 2, ri * 100 + 1.5,
                    f"{ri*100:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "fig02_impedance_mismatch.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_power_law_attenuation() -> None:
    tissues = {
        "Water (viscothermal)": (2.5e-4 / 11.51 * 100, 2.0),
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
        alpha = kw.absorption_power_law_db_cm(a0, y, f_MHz)
        ax.plot(f_MHz, alpha, label=name, color=color, linewidth=1.8)

    ax.set_xlabel("Frequency $f$ [MHz]", fontsize=12)
    ax.set_ylabel(r"Attenuation $\alpha(f)$ [dB cm$^{-1}$]", fontsize=12)
    ax.set_title(r"Figure 1.3 — Power-law attenuation $\alpha(f) = \alpha_0 f^y$ (Duck 1990)", fontsize=11)
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


def fig_harmonic_generation() -> None:
    f0 = 1e6
    p0 = 1e5
    x = np.linspace(0.0, 0.10, 500)

    # Water: beta=3.5, c=1481, rho=998
    p2_water = kw.fubini_harmonic_amplitude(3.5, 2 * math.pi * f0 / 1481.0, p0, 998.0, 1481.0, x)
    # Soft tissue: beta=4.7, c=1540, rho=1060
    p2_tissue = kw.fubini_harmonic_amplitude(4.7, 2 * math.pi * f0 / 1540.0, p0, 1060.0, 1540.0, x)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(x * 1e2, p2_water / 1e3, label=r"Water ($\beta=3.5$)", linewidth=2.0)
    ax.plot(x * 1e2, p2_tissue / 1e3, label=r"Soft tissue ($\beta=4.7$)", linewidth=2.0, linestyle="--")
    ax.set_xlabel("Propagation distance $x$ [cm]", fontsize=12)
    ax.set_ylabel("Second harmonic amplitude $p_2$ [kPa]", fontsize=12)
    ax.set_title(
        "Figure 1.4 — Linear growth of the second harmonic (Theorem 1.8)\n"
        r"$p_2(x) \approx \frac{\beta k p_0^2}{4 \rho_0 c_0^3}\,x$",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = OUT_DIR / "fig04_harmonic_generation.pdf"
    fig.savefig(path, dpi=FIG_DPI)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_sound_speed_temperature() -> None:
    T = np.linspace(0, 80, 500)
    c = kw.water_sound_speed_temperature(T)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(T, c, linewidth=2.0, color="#1f77b4")
    ax.axvline(20, color="k", linestyle=":", alpha=0.6)
    ax.axvline(37, color="#d62728", linestyle=":", alpha=0.6)
    ax.text(21, 1450, "20 degC", fontsize=9, color="k")
    ax.text(38, 1450, "37 degC\n(body temperature)", fontsize=9, color="#d62728")
    ax.set_xlabel("Temperature $T$ [degC]", fontsize=12)
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


def main() -> None:
    print(f"\nChapter 1 figures -> {OUT_DIR}\n")
    fig_standing_wave()
    fig_impedance_mismatch()
    fig_power_law_attenuation()
    fig_harmonic_generation()
    fig_sound_speed_temperature()
    print("\nDone.")


if __name__ == "__main__":
    main()
