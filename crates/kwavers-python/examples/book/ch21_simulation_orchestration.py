"""
Chapter 21 figure — Simulation Orchestration (§21.8.3)
======================================================

Bubble radius dynamics under the three kwavers bubble-dynamics solvers
(Rayleigh-Plesset, Keller-Miksis, Gilmore-Tait) for a 5-µm air bubble in water
driven by a 1-MHz, 200-kPa sinusoid.

All physics is in the Rust core via pykwavers — the three ODEs are integrated by
kw.solve_rayleigh_plesset / kw.solve_keller_miksis / kw.solve_gilmore (the same
solvers the plugin catalog dispatches to). Only the drive set-up and plotting
are in Python.

Output: docs/book/figures/ch21sim/fig01_bubble_ode_comparison.{pdf,png}

References
----------
Brennen C.E. (1995) Cavitation and Bubble Dynamics, Oxford UP, §4.2.
Gilmore F.R. (1952) Caltech Hydrodynamics Lab Report 26-4.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21sim")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.4,
})

# Physical set-up: 5-µm air bubble in water, 1 MHz, 200 kPa.
R0 = 5e-6        # equilibrium radius [m]
RHO = 1000.0     # liquid density [kg/m³]
C0 = 1500.0      # liquid sound speed [m/s]
P0 = 101325.0    # ambient pressure [Pa]
SIGMA = 0.0725   # surface tension [N/m]
MU = 1.0e-3      # dynamic viscosity [Pa·s]
GAMMA = 1.4      # air adiabatic index
PV = 0.0         # vapour pressure (Pa); 0 → non-condensable gas only
F0 = 1.0e6       # drive frequency [Hz]
PA = 200_000.0   # drive amplitude [Pa]

T_END = 4.0 / F0          # 4 acoustic periods
N_STEPS = 8000            # 2000 steps/period


def fig01_bubble_ode_comparison() -> None:
    t_rp, r_rp, _ = kw.solve_rayleigh_plesset(
        R0, 0.0, P0, PA, F0, T_END, N_STEPS, RHO, SIGMA, GAMMA, MU, PV)
    t_km, r_km, _ = kw.solve_keller_miksis(
        R0, 0.0, P0, PA, F0, T_END, N_STEPS, RHO, SIGMA, GAMMA, MU, PV, C0)
    t_gil, r_gil, rdot_gil = kw.solve_gilmore(
        R0, 0.0, P0, PA, F0, T_END, N_STEPS, RHO, SIGMA, GAMMA, MU, PV, C0)

    t_us = np.asarray(t_rp) * 1e6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(t_us, np.asarray(r_rp) / R0, label="Rayleigh–Plesset", color="#1f77b4")
    ax1.plot(t_us, np.asarray(r_km) / R0, label="Keller–Miksis", color="#ff7f0e")
    ax1.plot(np.asarray(t_gil) * 1e6, np.asarray(r_gil) / R0, "--",
             label="Gilmore–Tait", color="#2ca02c")
    ax1.axhline(1.0, color="gray", lw=0.5)
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel(r"Normalised radius $R/R_0$")
    ax1.set_title(r"Bubble response — $R_0=5\,\mu$m, 1 MHz, 200 kPa")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.asarray(t_gil) * 1e6, np.asarray(rdot_gil), color="#2ca02c")
    ax2.axhline(0.0, color="gray", lw=0.5)
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel(r"Wall velocity $\dot R$ (m/s)")
    ax2.set_title("Gilmore wall velocity (inertial collapse/rebound)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Bubble radius dynamics — three kwavers solvers (§21.8.3)", y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"fig01_bubble_ode_comparison.{ext}"),
                    dpi=150, bbox_inches="tight")
    print("  saved: docs/book/figures/ch21sim/fig01_bubble_ode_comparison.{pdf,png}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 21 figure (bubble ODE comparison)...")
    fig01_bubble_ode_comparison()
    print("Done. Output: docs/book/figures/ch21sim/")
