"""
Chapter 16 figure generation — Transcranial Ultrasound
=======================================================

Produces publication-quality figures for docs/book/transcranial_ultrasound.md.

Output directory: docs/book/figures/ch16/

Figures produced
----------------
fig01  Skull insertion loss vs frequency (power-law attenuation model)
fig02  Phase aberration: wavefront distortion by heterogeneous skull
fig03  CT-to-acoustic property conversion: HU→c and HU→ρ
fig04  Transcranial focusing gain vs skull phase aberration (σ_φ)
fig05  Safety: skull surface temperature rise vs HIFU exposure time

References
----------
Aubry et al. (2003) J. Acoust. Soc. Am. 113(1):84
Clement & Hynynen (2002) Phys. Med. Biol. 47(8):1219
Schneider et al. (1996) Phys. Med. Biol. 41:111
Maréchal (1947) Rev. d'Optique 26:257
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch16")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch16/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})


# ── Figure 01: Skull insertion loss vs frequency ──────────────────────────────
def fig01_skull_insertion_loss() -> None:
    """
    Skull attenuation: α_skull ≈ 5–12 dB/cm at 1 MHz (Fry & Barger 1978).
    Two-way insertion loss (through and back) for different skull thicknesses.
    α(f) = α₀ f^1.2  [dB/cm]
    """
    f_MHz = np.linspace(0.25, 2.0, 300)
    alpha0 = 6.0  # dB/(cm·MHz^1.2) — typical calvaria

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for L_mm, col, lbl in [(3, "#1f77b4", "3 mm (thin)"), (6, "#ff7f0e", "6 mm (typical)"),
                            (9, "#2ca02c", "9 mm (thick)")]:
        L_cm = L_mm / 10
        IL = alpha0 * f_MHz**1.2 * 2 * L_cm  # two-way dB
        ax.plot(f_MHz, IL, color=col, label=lbl)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Two-way insertion loss (dB)")
    ax.set_title(r"Skull insertion loss: $\alpha(f) = \alpha_0 f^{1.2}$, two-way")
    ax.legend(title="Skull thickness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig01_skull_insertion_loss")
    plt.close(fig)


# ── Figure 02: Phase aberration distortion ────────────────────────────────────
def fig02_phase_aberration() -> None:
    """
    Demonstrate effect of skull phase aberration on wavefront.
    Incident plane wave + random phase screen → distorted wavefront.
    Corrected (time-reversed) → restored phase.
    """
    N = 128
    x = np.linspace(-30, 30, N)   # mm

    rng = np.random.default_rng(42)
    sigma_phi = 3.0  # radians — typical skull RMS phase aberration

    ideal = np.ones(N)  # flat wavefront
    aberrated_phase = sigma_phi * rng.standard_normal(N)
    aberrated = np.exp(1j * aberrated_phase)
    corrected = aberrated * np.exp(-1j * aberrated_phase)  # perfect correction

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(x, np.real(ideal), color="#1f77b4")
    axes[0].set_ylabel("Re(wavefront)")
    axes[0].set_title("Ideal flat wavefront (no skull)")
    axes[0].set_ylim(-2, 2)

    axes[1].plot(x, np.real(aberrated), color="#d62728")
    axes[1].set_ylabel("Re(wavefront)")
    axes[1].set_title(f"After skull aberration ($\\sigma_\\phi = {sigma_phi:.0f}$ rad)")
    axes[1].set_ylim(-2, 2)

    axes[2].plot(x, np.real(corrected), color="#2ca02c")
    axes[2].set_ylabel("Re(wavefront)")
    axes[2].set_title("After time-reversal phase conjugation (corrected)")
    axes[2].set_ylim(-2, 2)
    axes[2].set_xlabel("Transducer element position (mm)")

    fig.tight_layout()
    savefig("fig02_phase_aberration")
    plt.close(fig)


# ── Figure 03: CT to acoustic property conversion ────────────────────────────
def fig03_ct_conversion() -> None:
    """
    Schneider (1996) HU → ρ → c conversion:
    ρ(HU) = 1.0 + HU/1000  [g/cm³]    (water calibration)
    c(HU) = 1500 + 0.79·HU^0.88  [m/s] for HU > 0 (simplified)
    c(HU) = 1500 + 0.79·HU        for HU ≤ 0
    """
    HU = np.linspace(-1000, 3000, 500)

    # Density (linear with HU, water at 0 HU)
    rho = 1.0 + HU / 1000.0  # g/cm³
    rho = np.clip(rho, 0.001, 3.0)

    # Speed of sound
    c = np.where(HU <= 0,
                 1500 + 0.79 * HU,
                 1500 + 0.79 * np.clip(HU, 0, None)**0.88)
    c = np.clip(c, 300, 4500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(HU, rho, color="#1f77b4")
    ax1.axvline(0, color="k", linewidth=0.5, linestyle=":")
    ax1.axvline(1000, color="gray", linewidth=0.5, linestyle="--", label="HU=1000 (cortical bone)")
    ax1.set_xlabel("CT number (HU)")
    ax1.set_ylabel(r"Density $\rho$ (g/cm³)")
    ax1.set_title("HU → density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(HU, c, color="#d62728")
    ax2.axvline(0, color="k", linewidth=0.5, linestyle=":")
    ax2.axvline(1000, color="gray", linewidth=0.5, linestyle="--", label="HU=1000")
    ax2.axhline(1500, color="blue", linewidth=0.5, linestyle=":", label="c = 1500 m/s (water)")
    ax2.set_xlabel("CT number (HU)")
    ax2.set_ylabel("Sound speed $c$ (m/s)")
    ax2.set_title("HU → speed of sound (Schneider 1996)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig03_ct_conversion")
    plt.close(fig)


# ── Figure 04: Focusing gain vs phase aberration ──────────────────────────────
def fig04_strehl_ratio() -> None:
    """
    Maréchal approximation: Strehl ratio S ≈ exp(-σ_φ²)
    (exact for Gaussian phase distribution and small aberrations).
    Focus intensity normalized to aberration-free.
    """
    sigma_phi = np.linspace(0, 4.0, 300)
    S = np.exp(-sigma_phi**2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sigma_phi, S, color="#1f77b4")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="S = 0.5 (3 dB loss)")
    ax.axhline(0.2, color="orange", linestyle=":", linewidth=1, label="S = 0.2 (7 dB loss)")
    # Mark typical skull σ_φ
    sigma_skull = [1.5, 2.5, 3.5]
    for s in sigma_skull:
        ax.axvline(s, color="gray", linewidth=0.5, linestyle="-.")

    ax.set_xlabel(r"RMS phase aberration $\sigma_\phi$ (rad)")
    ax.set_ylabel(r"Strehl ratio $S = e^{-\sigma_\phi^2}$")
    ax.set_title("Transcranial focus degradation (Maréchal approximation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    savefig("fig04_strehl_ratio")
    plt.close(fig)


# ── Figure 05: Skull surface temperature rise ────────────────────────────────
def fig05_skull_temperature() -> None:
    """
    Simplified skull heating: ΔT ≈ α_skull · I · t / (ρ_skull · C_skull)
    where I = ISPTA, α_skull = absorption coefficient (Np/m), t = exposure.
    Two contributions: skull surface (high absorption) vs brain target (low).
    """
    t = np.linspace(0, 30, 300)   # seconds

    # Skull surface (bone, high absorption)
    alpha_skull = 40.0   # Np/m at 1 MHz
    I_skull = 1e4        # W/m² = 1 W/cm² at skull surface
    rho_skull = 1900.0
    C_skull = 1300.0
    dT_skull = alpha_skull * I_skull * t / (rho_skull * C_skull)

    # Brain tissue target (much lower absorption)
    alpha_brain = 1.0    # Np/m
    I_brain = 500.0      # W/m² at focus
    rho_brain = 1040.0
    C_brain = 3600.0
    dT_brain = alpha_brain * I_brain * t / (rho_brain * C_brain)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, dT_skull, color="#d62728", label="Skull surface (cortical bone)")
    ax.plot(t, dT_brain, color="#1f77b4", linestyle="--", label="Brain target (soft tissue)")
    ax.axhline(1.0, color="orange", linestyle=":", linewidth=1.5, label="ΔT = 1°C (guidance)")
    ax.axhline(6.0, color="r", linestyle="--", linewidth=1.5, label="ΔT = 6°C (damage threshold)")
    ax.set_xlabel("Exposure time $t$ (s)")
    ax.set_ylabel("Temperature rise $\\Delta T$ (°C)")
    ax.set_title("Transcranial HIFU: skull vs brain temperature rise\n"
                 r"$\Delta T = \alpha I t / (\rho C)$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    fig.tight_layout()
    savefig("fig05_skull_temperature")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 16 figures (Transcranial Ultrasound)...")
    fig01_skull_insertion_loss()
    fig02_phase_aberration()
    fig03_ct_conversion()
    fig04_strehl_ratio()
    fig05_skull_temperature()
    print("Done. Output: docs/book/figures/ch16/")
