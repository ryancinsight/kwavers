"""
Chapter 15 figure generation — Safety and Dosimetry
====================================================

Produces publication-quality figures for docs/book/safety_and_dosimetry.md.
Figures derive from FDA/IEC safety indices (MI, TI) and CEM43 thermal dose.

Output directory: docs/book/figures/ch15/

Figures produced
----------------
fig01  Mechanical index MI = p_neg / sqrt(f) vs frequency and pressure
fig02  Thermal index TI vs intensity and tissue depth
fig03  CEM43 thermal dose accumulation vs temperature
fig04  Arrhenius damage integral vs exposure time at 3 temperatures
fig05  FDA safety parameter space: ISPTA, ISPPA, MI limits

References
----------
FDA (2019) Guidance for Ultrasonic Diagnostic Systems
WFUMB (2019) Safety Statements for Diagnostic Ultrasound
IEC 62359:2010 Ultrasonics — Field characterization
Sapareto & Dewey (1984) Int. J. Radiat. Oncol. 10:787
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch15")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch15/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})


# ── Figure 01: Mechanical index ───────────────────────────────────────────────
def fig01_mechanical_index() -> None:
    """
    MI = p_neg [MPa] / sqrt(f [MHz])
    FDA limit: MI ≤ 1.9 (general imaging).
    Safe threshold for stable cavitation: MI < 0.3.
    """
    f_MHz = np.logspace(-1, 1, 300)  # 0.1–10 MHz
    p_neg_MPa = np.array([0.1, 0.3, 1.0, 1.9, 3.0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(p_neg_MPa)))
    for p, col in zip(p_neg_MPa, colors):
        MI = p / np.sqrt(f_MHz)
        ax.semilogx(f_MHz, MI, color=col, label=f"$p_- = {p:.1f}$ MPa")

    ax.axhline(1.9, color="r", linestyle="--", linewidth=1.5, label="FDA limit MI = 1.9")
    ax.axhline(0.3, color="orange", linestyle=":", linewidth=1.5, label="Cavitation threshold MI = 0.3")
    ax.fill_between(f_MHz, 0, 0.3, alpha=0.1, color="green", label="Safe zone")
    ax.fill_between(f_MHz, 1.9, 4.0, alpha=0.1, color="red")
    ax.set_xlabel("Frequency $f$ (MHz)")
    ax.set_ylabel(r"Mechanical Index $\mathrm{MI} = p_-/\sqrt{f}$")
    ax.set_title(r"Mechanical Index: $\mathrm{MI} = p_-\,[\mathrm{MPa}]\,/\sqrt{f\,[\mathrm{MHz}]}$")
    ax.legend(fontsize=8)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 4)
    fig.tight_layout()
    savefig("fig01_mechanical_index")
    plt.close(fig)


# ── Figure 02: Thermal index ──────────────────────────────────────────────────
def fig02_thermal_index() -> None:
    """
    TIS (soft tissue) = W_0 · f / (40 · A_aprt)
    TIB (bone at focus) = W_3 · f^(1/4) / 40  (simplified FDA formula)
    Show TIS vs ISPTA.3 for typical imaging modes.
    ISPTA.3 in mW/cm² (derated at 0.3 dB/cm/MHz).
    """
    ISPTA_mWcm2 = np.linspace(0, 1000, 400)

    # TIS approximation: TIS ≈ ISPTA.3 / 210 (Thomenius rule of thumb)
    TIS = ISPTA_mWcm2 / 210.0

    # TIB: higher for bone (factor 3×)
    TIB = TIS * 3.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ISPTA_mWcm2, TIS, label="TIS (soft tissue)")
    ax.plot(ISPTA_mWcm2, TIB, "--", label="TIB (bone at focus)")
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1.5, label="TI = 1.0 (guidance level)")
    ax.axhline(3.0, color="darkred", linestyle=":", linewidth=1.5, label="TI = 3.0 (neonatal cranium)")
    ax.axvline(720, color="gray", linestyle=":", linewidth=1, label="ISPTA.3 = 720 mW/cm² (FDA)")
    ax.fill_between(ISPTA_mWcm2, 0, np.minimum(TIS, 1.0), alpha=0.1, color="green")
    ax.set_xlabel(r"$I_{\mathrm{SPTA.3}}$ (mW/cm²)")
    ax.set_ylabel("Thermal Index (TI)")
    ax.set_title("Thermal Index vs spatial-peak temporal-average intensity")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 5)
    fig.tight_layout()
    savefig("fig02_thermal_index")
    plt.close(fig)


# ── Figure 03: CEM43 thermal dose ────────────────────────────────────────────
def fig03_cem43() -> None:
    """
    CEM43 = Σ R^(43-T) · Δt
    R = 0.25 for T < 43 °C, R = 0.5 for T ≥ 43 °C.
    Show CEM43 accumulation over time for step-function heating.
    """
    t_s = np.linspace(0, 300, 3000)  # 5 min exposure
    T_vals = [(41.0, "#1f77b4", "41°C"), (43.0, "#ff7f0e", "43°C"),
              (45.0, "#2ca02c", "45°C"), (50.0, "#d62728", "50°C")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for T, col, lbl in T_vals:
        R = 0.5 if T >= 43.0 else 0.25
        cem43_per_sec = R ** (43.0 - T)
        CEM43 = cem43_per_sec * t_s
        ax.semilogy(t_s / 60, CEM43 + 1e-12, color=col, label=lbl)

    ax.axhline(240, color="r", linestyle="--", linewidth=1.5, label="CEM43=240 min (necrosis threshold)")
    ax.axhline(1, color="orange", linestyle=":", linewidth=1, label="CEM43=1 min (reversible)")
    ax.set_xlabel("Exposure time (min)")
    ax.set_ylabel("CEM43 (min)")
    ax.set_title(r"CEM43 thermal dose: $\mathrm{CEM43} = \int R^{43-T}\,dt$")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 5)
    fig.tight_layout()
    savefig("fig03_cem43")
    plt.close(fig)


# ── Figure 04: Arrhenius damage integral ─────────────────────────────────────
def fig04_arrhenius_damage() -> None:
    """
    Ω(t) = ∫₀ᵗ A exp(-E_a / (R_g T(τ))) dτ
    Ω = 1: threshold for irreversible thermal damage.
    Typical values: A = 3.1e98 s⁻¹, E_a = 6.28e5 J/mol, R_g = 8.314 J/(mol·K)
    [Henriques & Moritz 1947, tissue].
    """
    A_coeff = 3.1e98    # s⁻¹
    Ea = 6.28e5         # J/mol
    Rg = 8.314          # J/(mol·K)

    t_max_s = 100.0
    t = np.linspace(0, t_max_s, 10000)
    dt = t[1] - t[0]

    T_celsius = [55.0, 60.0, 70.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for T_c, col in zip(T_celsius, colors):
        T_K = T_c + 273.15
        rate = A_coeff * np.exp(-Ea / (Rg * T_K))
        Omega = rate * t  # constant T → linear accumulation
        ax.semilogy(t, Omega, color=col, label=f"$T = {T_c:.0f}°C$")
        # Mark time when Omega = 1
        if rate > 0:
            t_threshold = 1.0 / rate
            if t_threshold < t_max_s:
                ax.scatter(t_threshold, 1.0, s=80, color=col, zorder=5)

    ax.axhline(1.0, color="r", linestyle="--", linewidth=1.5, label=r"$\Omega=1$ (damage threshold)")
    ax.set_xlabel("Exposure time $t$ (s)")
    ax.set_ylabel(r"Arrhenius damage integral $\Omega$")
    ax.set_title(r"Arrhenius thermal damage: $\Omega = A e^{-E_a/R_gT}\cdot t$")
    ax.legend()
    ax.set_xlim(0, t_max_s)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig04_arrhenius_damage")
    plt.close(fig)


# ── Figure 05: FDA safety parameter space ────────────────────────────────────
def fig05_fda_parameter_space() -> None:
    """
    FDA Output Display Standard (ODS) limits:
    - ISPTA.3 ≤ 720 mW/cm²  (all modes except ophthalmology)
    - ISPPA.3 ≤ 190 W/cm²   (all modes except ophthalmology)
    - MI ≤ 1.9
    Plot ISPTA.3 vs ISPPA.3 with FDA limit boundaries.
    Representative operating points for common imaging modes.
    """
    ISPPA = np.logspace(0, 3, 300)   # W/cm²

    fig, ax = plt.subplots(figsize=(8, 5))

    # FDA limits
    ax.axhline(720, color="r", linestyle="--", linewidth=1.5, label=r"$I_\mathrm{SPTA.3} = 720\,\mathrm{mW/cm^2}$")
    ax.axvline(190, color="b", linestyle="--", linewidth=1.5, label=r"$I_\mathrm{SPPA.3} = 190\,\mathrm{W/cm^2}$")

    # Shade allowed region
    ax.fill_between([1, 190], [0, 0], [720, 720], alpha=0.1, color="green")
    ax.text(5, 100, "FDA\nallowed\nregion", fontsize=9, color="green")

    # Representative operating points
    modes = [
        ("B-mode imaging", 15, 50, "#1f77b4", "o"),
        ("Color Doppler", 80, 150, "#ff7f0e", "s"),
        ("Pulsed Doppler", 400, 100, "#2ca02c", "^"),
        ("HIFU therapy", 9000, 600, "#d62728", "D"),
        ("Lithotripsy", 15000, 300, "#9467bd", "v"),
    ]

    for name, isppa, ispta, col, mk in modes:
        ax.scatter(isppa, ispta, s=100, color=col, marker=mk, zorder=5, label=name)

    ax.set_xlabel(r"$I_\mathrm{SPPA.3}$ (W/cm²)")
    ax.set_ylabel(r"$I_\mathrm{SPTA.3}$ (mW/cm²)")
    ax.set_title("FDA ultrasound safety parameter space")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1, 1e4)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig05_fda_parameter_space")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 15 figures (Safety and Dosimetry)...")
    fig01_mechanical_index()
    fig02_thermal_index()
    fig03_cem43()
    fig04_arrhenius_damage()
    fig05_fda_parameter_space()
    print("Done. Output: docs/book/figures/ch15/")
