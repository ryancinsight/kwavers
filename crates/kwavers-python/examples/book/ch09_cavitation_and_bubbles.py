"""
Chapter 9 figure generation — Cavitation and Bubble Dynamics
=============================================================

Produces publication-quality figures for docs/book/cavitation_and_bubbles.md.
All physics computed by kwavers (Rust); this file contains only matplotlib
rendering.  Requires pykwavers to be installed.

Output directory: docs/book/figures/ch09/

References
----------
Rayleigh (1917) Phil. Mag. 34:94
Plesset & Prosperetti (1977) Ann. Rev. Fluid Mech. 9:145
Minnaert (1933) Phil. Mag. 16:235
Blake (1949) Tech. Mem. 12, Acoustics Research Lab.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch09")
os.makedirs(OUT_DIR, exist_ok=True)

RHO = 998.0
SIGMA = 0.0728
P0_ATM = 101_325.0
PV = 2_338.0
GAMMA = 1.4
MU = 1.0e-3  # water dynamic viscosity [Pa·s]


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch09/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})


def fig01_rp_dynamics() -> None:
    f_us = 1.0e6
    R0 = 2.0e-6
    t_end = 5e-6
    n_steps = 5000

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for P_ac, lbl, ls in [
        (0.3e6, r"$P_{ac}=0.3\,\mathrm{MPa}$ (stable)", "solid"),
        (1.5e6, r"$P_{ac}=1.5\,\mathrm{MPa}$ (inertial)", "dashed"),
    ]:
        t, R, _ = kw.solve_rayleigh_plesset(
            R0, 0.0, P0_ATM, P_ac, f_us, t_end, n_steps, RHO, SIGMA, GAMMA, MU, PV
        )
        axes[0].plot(t * 1e6, R / R0, linestyle=ls, label=lbl)

    axes[0].set_xlabel(r"Time $t$ (us)")
    axes[0].set_ylabel(r"$R(t) / R_0$")
    axes[0].set_title(r"Rayleigh-Plesset dynamics, $R_0 = 2\,\mu\mathrm{m}$, $f = 1\,\mathrm{MHz}$")
    axes[0].legend()
    axes[0].axhline(1.0, color="k", linewidth=0.5, linestyle=":")

    t2, R2, _ = kw.solve_rayleigh_plesset(
        R0, 0.0, P0_ATM, 1.5e6, f_us, 1.5e-6, 3000, RHO, SIGMA, GAMMA, MU, PV
    )
    axes[1].plot(t2 * 1e6, R2 / R0, color="#d62728")
    axes[1].set_xlabel(r"Time $t$ (us)")
    axes[1].set_ylabel(r"$R(t) / R_0$")
    axes[1].set_title(r"Inertial collapse detail ($P_{ac}=1.5\,\mathrm{MPa}$)")
    axes[1].set_yscale("log")

    fig.tight_layout()
    savefig("fig01_rp_dynamics")
    plt.close(fig)


def fig02_minnaert_resonance() -> None:
    R0_arr = np.logspace(-7, -3, 400)
    f_M = np.vectorize(lambda r: kw.minnaert_resonance_hz(r, GAMMA, P0_ATM, RHO))(R0_arr)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(R0_arr * 1e6, f_M * 1e-6, color="#1f77b4")
    R0_ref = np.logspace(-7, -3, 5)
    f_ref = 3.26 / R0_ref
    ax.loglog(R0_ref * 1e6, f_ref * 1e-6, "k--", linewidth=1,
              label=r"$f_M \approx 3.26/R_0$ (Minnaert approx.)")
    ax.set_xlabel(r"Equilibrium radius $R_0$ (um)")
    ax.set_ylabel(r"Minnaert frequency $f_M$ (MHz)")
    ax.set_title(r"Minnaert resonance: $f_M = \frac{1}{2\pi R_0}\sqrt{\frac{3\gamma P_0}{\rho}}$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.1, 1000)
    fig.tight_layout()
    savefig("fig02_minnaert_resonance")
    plt.close(fig)


def fig03_blake_threshold() -> None:
    R0_arr = np.logspace(np.log10(0.1e-6), np.log10(100e-6), 500)
    P_B = np.vectorize(lambda r: kw.blake_threshold_pa(r, P0_ATM, SIGMA))(R0_arr)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(R0_arr * 1e6, P_B * 1e-6, color="#d62728")
    ax.set_xlabel(r"Equilibrium radius $R_0$ (um)")
    ax.set_ylabel(r"Blake threshold $|P_B|$ (MPa)")
    ax.set_title("Blake inertial cavitation threshold vs bubble radius")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1, label="0.1 MPa reference")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="1 MPa reference")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig03_blake_threshold")
    plt.close(fig)


def fig04_collapse_time() -> None:
    R_max = np.logspace(np.log10(1e-6), np.log10(1e-3), 300)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for delta_P, lbl, col in [
        (P0_ATM - PV, r"$\Delta P = P_0 - P_v$ (atmospheric)", "#1f77b4"),
        (5e5, r"$\Delta P = 0.5\,\mathrm{MPa}$", "#ff7f0e"),
        (2e6, r"$\Delta P = 2\,\mathrm{MPa}$", "#2ca02c"),
    ]:
        t_c = np.vectorize(lambda rm: kw.rayleigh_collapse_time_s(rm, delta_P, RHO))(R_max)
        ax.loglog(R_max * 1e6, t_c * 1e9, label=lbl, color=col)

    ax.set_xlabel(r"Maximum radius $R_\mathrm{max}$ (um)")
    ax.set_ylabel(r"Rayleigh collapse time $t_c$ (ns)")
    ax.set_title(r"Rayleigh collapse time: $t_c = 0.9147\,R_\mathrm{max}\sqrt{\rho/\Delta P}$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig04_collapse_time")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 9 figures (Cavitation and Bubble Dynamics)...")
    fig01_rp_dynamics()
    fig02_minnaert_resonance()
    fig03_blake_threshold()
    fig04_collapse_time()
    print("Done. Output: docs/book/figures/ch09/")
