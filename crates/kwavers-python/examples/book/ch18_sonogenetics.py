"""
Chapter 18 figure generation — Sonogenetics
============================================

Produces publication-quality figures for docs/book/sonogenetics.md.

Output directory: docs/book/figures/ch18/

Figures produced
----------------
fig01  Boltzmann channel gating: P_open vs membrane tension/pressure
fig02  Acoustic radiation force on a cell vs cell radius and frequency
fig03  Streaming shear stress vs transducer intensity
fig04  Safety: CEM43 thermal dose for typical sonogenetics parameters
fig05  Neural activation threshold: sonogenetics vs optogenetics comparison

References
----------
Ibsen et al. (2015) Nature Commun. 6:8264
Yosioka & Kawasima (1955) Acustica 5:167
Legon et al. (2014) Nature Neurosci. 17:322
Sapareto & Dewey (1984) Int. J. Radiat. Oncol. 10:787
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch18")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch18/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

kB = 1.38e-23  # J/K
T_body = 310.0  # K (37°C)


# ── Figure 01: Boltzmann channel gating ──────────────────────────────────────
def fig01_channel_gating() -> None:
    """
    Two-state Boltzmann model:
    P_open(F) = 1 / (1 + exp(-(F - F_0) / (kB T)))
    where F = ΔP · A_m (force on membrane patch of area A_m).
    Calibrated to MscL-G22S (Ibsen 2015): F_0 ≈ 15 pN, kBT ≈ 4.28 pN·nm.
    """
    # Parameters for three channels (representative)
    channels = [
        ("MscL-G22S (low threshold)", 5.0, 2.0, "#1f77b4"),
        ("MscL wildtype", 15.0, 3.0, "#ff7f0e"),
        ("Piezo1", 30.0, 5.0, "#2ca02c"),
    ]

    F_pN = np.linspace(-10, 60, 500)  # pN

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, F0, kT, col in channels:
        P_open = 1.0 / (1.0 + np.exp(-(F_pN - F0) / kT))
        ax.plot(F_pN, P_open, color=col, label=name)

    ax.axhline(0.5, color="k", linewidth=0.5, linestyle="--", label=r"$P_\mathrm{open}=0.5$")
    ax.set_xlabel("Membrane force $F$ (pN)")
    ax.set_ylabel("Open probability $P_\\mathrm{open}$")
    ax.set_title(r"Boltzmann channel gating: $P_\mathrm{open}(F) = [1 + e^{-(F-F_0)/k_BT}]^{-1}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 60)
    fig.tight_layout()
    savefig("fig01_channel_gating")
    plt.close(fig)


# ── Figure 02: Acoustic radiation force on a cell ────────────────────────────
def fig02_radiation_force() -> None:
    """
    Primary radiation force (Yosioka-Kawasima 1955, Gorkov 1962):
    F_rad ∝ (2π r³/3c) · [f₁ κ̃ - (3/2) f₂ ρ̃] · d<p²>/dx
    For a cell in a standing wave, F ~ r³ P²/c.
    Show F_rad vs cell radius (r = 1–20 µm) and vs frequency.
    """
    P_rms = 5e4   # Pa (50 kPa)
    c = 1500.0
    kappa_water = 1.0  # relative
    rho_cell = 1.07    # normalised density (relative to water)
    kappa_cell = 0.8   # relative compressibility

    # Gorkov potential acoustic contrast factor
    f1 = 1 - kappa_cell / kappa_water
    f2 = 2 * (rho_cell - 1) / (2 * rho_cell + 1)
    Phi = (f1 / 3) - f2 / 2  # acoustic contrast factor

    r_um = np.linspace(0.5, 20, 200)
    r_m = r_um * 1e-6
    F_pN = abs(Phi) * 4 * np.pi * r_m**3 * P_rms**2 / (3 * c * 1480) * 1e12  # pN

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(r_um, F_pN, color="#1f77b4")
    ax1.set_xlabel(r"Cell radius $r$ (µm)")
    ax1.set_ylabel(r"Radiation force $F_\mathrm{rad}$ (pN)")
    ax1.set_title(r"ARF vs cell radius" "\n" r"($P_\mathrm{rms}=50$ kPa, $f=1$ MHz)")
    ax1.grid(True, alpha=0.3)

    # vs pressure (at fixed r = 10 µm)
    P_arr = np.linspace(1e3, 5e5, 300)  # 1 kPa to 500 kPa
    r_fixed = 10e-6
    F_pN2 = abs(Phi) * 4 * np.pi * r_fixed**3 * P_arr**2 / (3 * c * 1480) * 1e12
    ax2.plot(P_arr * 1e-3, F_pN2, color="#d62728")
    ax2.set_xlabel("Peak pressure $P$ (kPa)")
    ax2.set_ylabel(r"Radiation force $F_\mathrm{rad}$ (pN)")
    ax2.set_title(r"ARF vs pressure ($r=10\,\mu\mathrm{m}$)")
    # Activation threshold reference
    ax2.axhline(5.0, color="k", linestyle="--", linewidth=1, label="MscL threshold ~5 pN")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig02_radiation_force")
    plt.close(fig)


# ── Figure 03: Streaming shear stress ────────────────────────────────────────
def fig03_streaming_shear() -> None:
    """
    Acoustic streaming velocity: v_s = I α / (2 ρ c² f)  [Nyborg 1953]
    Shear stress on membrane: τ = η v_s / δ_BL  where δ_BL is boundary layer.
    """
    I_Wcm2 = np.logspace(-1, 3, 300)  # 0.1 to 1000 W/cm²
    I_Wm2 = I_Wcm2 * 1e4

    alpha = 1.0    # Np/m at 1 MHz (water)
    rho = 998.0
    c = 1500.0
    f = 1e6
    eta = 1e-3     # Pa·s dynamic viscosity
    delta_BL = 0.56e-6  # boundary layer ~0.56 µm at 1 MHz

    v_s = I_Wm2 * alpha / (2 * rho * c**2)  # but this is Eckart streaming...
    # More standard: v_s = 2 alpha I / (rho omega)
    omega = 2 * np.pi * f
    v_s_eckart = 2 * alpha * I_Wm2 / (rho * omega)

    tau = eta * v_s_eckart / delta_BL  # Pa

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(I_Wcm2, v_s_eckart * 1e6, color="#1f77b4")
    ax1.set_xlabel(r"Intensity $I$ (W/cm²)")
    ax1.set_ylabel(r"Streaming velocity $v_s$ (µm/s)")
    ax1.set_title(r"Acoustic streaming: $v_s = 2\alpha I/(\rho\omega)$")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.loglog(I_Wcm2, tau, color="#d62728")
    ax2.axhline(0.1, color="k", linestyle="--", linewidth=1, label="Permeabilisation threshold ~0.1 Pa")
    ax2.set_xlabel(r"Intensity $I$ (W/cm²)")
    ax2.set_ylabel(r"Shear stress $\tau$ (Pa)")
    ax2.set_title("Streaming shear stress on membrane")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    savefig("fig03_streaming_shear")
    plt.close(fig)


# ── Figure 04: CEM43 safety budget for sonogenetics ──────────────────────────
def fig04_safety_budget() -> None:
    """
    Typical LIFU sonogenetics parameters:
    f = 0.5 MHz, ISPPA = 5 W/cm², duty cycle DC ∈ [0.1%, 50%].
    Temperature rise: dT = 2α I DC / (2ρCp) [simplified steady-state].
    CEM43 dose via kw.cem43_at_temperatures(T_arr, t_on_s) [Rust kernel, min].
    t_on = t_stim · DC (on-time in seconds varies per DC).
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig04 (CEM43 safety budget)")
    DC = np.linspace(0.001, 0.5, 300)  # 0.1% to 50%
    alpha_tissue = 2.0  # Np/m at 0.5 MHz
    rho_b = 1040.0
    Cp_b = 3600.0
    I_vals = [1e4, 5e4, 1e5]   # W/m²  (1, 5, 10 W/cm²)
    t_stim = 30.0  # s total stimulation time

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for I_Wm2, col, lbl in zip(I_vals, ["#1f77b4", "#ff7f0e", "#2ca02c"],
                                [r"$I=1\,\mathrm{W/cm^2}$", r"$I=5\,\mathrm{W/cm^2}$",
                                 r"$I=10\,\mathrm{W/cm^2}$"]):
        dT = 2 * alpha_tissue * I_Wm2 * DC / (2 * rho_b * Cp_b)
        T_total = 37.0 + dT   # temperature [°C] per DC value
        # CEM43 dose [min] for each DC: temperature T_total[i], on-time t_stim*DC[i]
        CEM43 = np.array([
            float(np.asarray(kw.cem43_at_temperatures(np.array([T_total[i]]), t_stim * DC[i]))[0])
            for i in range(len(DC))
        ])
        ax.semilogy(DC * 100, CEM43 + 1e-15, color=col, label=lbl)

    ax.axhline(0.01, color="r", linestyle="--", linewidth=1.5, label="CEM43 = 0.01 min (safe)")
    ax.set_xlabel("Duty cycle (%)")
    ax.set_ylabel("CEM43 thermal dose (min)")
    ax.set_title("Sonogenetics thermal safety: CEM43 vs duty cycle")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig04_safety_budget")
    plt.close(fig)


# ── Figure 05: Activation threshold comparison ────────────────────────────────
def fig05_activation_comparison() -> None:
    """
    Compare activation parameters:
    - Optogenetics: μW/mm² light intensity, ms resolution
    - Sonogenetics: mW/cm² ultrasound, ms resolution
    Show spatial resolution vs energy per pulse for both.
    """
    # Representative data points (approximate from literature)
    methods = {
        "Optogenetics (ChR2)": {
            "spatial_um": 10,   # µm resolution
            "energy_nJ": 0.01,  # nJ/pulse
            "col": "#1f77b4", "mk": "o",
        },
        "Optogenetics (C1V1)": {
            "spatial_um": 10,
            "energy_nJ": 0.1,
            "col": "#ff7f0e", "mk": "o",
        },
        "Sonogenetics (MscL-G22S)": {
            "spatial_um": 1000,  # ~1 mm with focused US
            "energy_nJ": 1000,   # ~1 µJ
            "col": "#d62728", "mk": "^",
        },
        "Sonogenetics (TRPA1)": {
            "spatial_um": 2000,
            "energy_nJ": 5000,
            "col": "#9467bd", "mk": "^",
        },
        "Focused TUS (neuromod)": {
            "spatial_um": 5000,
            "energy_nJ": 50000,
            "col": "#8c564b", "mk": "s",
        },
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in methods.items():
        ax.scatter(d["energy_nJ"], d["spatial_um"],
                   s=120, color=d["col"], marker=d["mk"], zorder=5, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy per pulse (nJ)")
    ax.set_ylabel("Spatial resolution (µm)")
    ax.set_title("Activation threshold: sonogenetics vs optogenetics")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig05_activation_comparison")
    plt.close(fig)


# ── Figure 06: Sonogenetic transduction pipeline schematic ────────────────────
def fig06_pipeline_schematic() -> None:
    """Box-and-arrow schematic of the acoustic→mechanical→chemical→electrical chain."""
    stages = [
        ("Acoustic field\n$p(x,t)$", "PSTD / FDTD\nforward solve"),
        ("Radiation force\n$F=2\\alpha I/c$", "$VolumetricArfField$"),
        ("Membrane tension\n$\\Delta\\gamma=I R/2c$", "Laplace law\n$compute\\_membrane\\_tension$"),
        ("Channel gating\n$P_{open}(\\Delta\\gamma)$", "Boltzmann\n$boltzmann\\_p\\_open$"),
        ("Ion current\n$I_{ion}=gNAP_{open}\\Delta V$", "$ion\\_current$"),
        ("LIF spike\n$V_m(t)\\to$ AP", "$LifNeuron::step$"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    n = len(stages)
    box_w, box_h, gap = 1.55, 1.0, 0.45
    for i, (label, impl) in enumerate(stages):
        x = i * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, 0), box_w, box_h, fill=True,
                                   facecolor="#eaf2fb", edgecolor="#2b6cb0", lw=1.4))
        ax.text(x + box_w / 2, box_h * 0.62, label, ha="center", va="center", fontsize=9)
        ax.text(x + box_w / 2, box_h * 0.20, impl, ha="center", va="center",
                fontsize=7, style="italic", color="#555")
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, box_h / 2), xytext=(x + box_w, box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#c05621", lw=1.6))
    ax.set_xlim(-0.2, n * (box_w + gap))
    ax.set_ylim(-0.2, box_h + 0.2)
    ax.axis("off")
    ax.set_title("Sonogenetic transduction pipeline: acoustic → mechanical → chemical → electrical",
                 fontsize=11)
    fig.tight_layout()
    savefig("fig06_pipeline_schematic")
    plt.close(fig)


# ── Figure 07: LIF spike raster vs duty cycle ─────────────────────────────────
def fig07_lif_raster_vs_duty() -> None:
    """LIF spike raster across pulse duty cycles, driven by a pulsed sonogenetic current.

    The ion current is a square-pulse train (PRF fixed) whose ON fraction is the
    duty cycle. simulate_lif_neuron_py (Rust LIF, Koch 1999) integrates V_m(t) and
    returns spike times; higher duty cycle deposits more charge → more spikes.
    """
    if not _HAS_PYKWAVERS:
        print("  [skip fig07] pykwavers unavailable")
        return
    dt_s = 1.0e-4
    t_total = 1.0  # s
    n = int(t_total / dt_s)
    t = np.arange(n) * dt_s
    prf = 10.0  # Hz pulse-repetition frequency
    i_on = 4.0e-10  # A — supra-threshold on-pulse current
    duty_cycles = [0.02, 0.05, 0.10, 0.20, 0.50]

    fig, (ax_r, ax_n) = plt.subplots(1, 2, figsize=(12, 4.2),
                                     gridspec_kw={"width_ratios": [2.2, 1]})
    rates = []
    for row, dc in enumerate(duty_cycles):
        phase = (t * prf) % 1.0
        i_ion = np.where(phase < dc, i_on, 0.0).astype(np.float64)
        res = kw.simulate_lif_neuron_py(np.ascontiguousarray(i_ion), dt_s)
        spikes = np.asarray(res["spike_times_s"], dtype=np.float64)
        ax_r.vlines(spikes, row + 0.6, row + 1.4, color="#2b6cb0", lw=1.0)
        rates.append(spikes.size / t_total)
    ax_r.set_yticks(range(1, len(duty_cycles) + 1))
    ax_r.set_yticklabels([f"{int(dc*100)}%" for dc in duty_cycles])
    ax_r.set_xlabel("Time [s]")
    ax_r.set_ylabel("Duty cycle")
    ax_r.set_title("LIF spike raster vs duty cycle (10 Hz PRF, fixed on-current)")
    ax_r.set_xlim(0, t_total)
    ax_r.grid(True, axis="x", lw=0.3, alpha=0.5)

    ax_n.plot([dc * 100 for dc in duty_cycles], rates, "o-", color="#c05621")
    ax_n.set_xlabel("Duty cycle [%]")
    ax_n.set_ylabel("Firing rate [Hz]")
    ax_n.set_title("Rate vs duty cycle")
    ax_n.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    savefig("fig07_lif_raster_vs_duty")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 18 figures (Sonogenetics)...")
    fig01_channel_gating()
    fig02_radiation_force()
    fig03_streaming_shear()
    fig04_safety_budget()
    fig05_activation_comparison()
    fig06_pipeline_schematic()
    fig07_lif_raster_vs_duty()
    print("Done. Output: docs/book/figures/ch18/")
