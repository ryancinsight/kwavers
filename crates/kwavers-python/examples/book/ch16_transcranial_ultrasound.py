"""
Chapter 16 figure generation — Transcranial Ultrasound
=======================================================

Produces publication-quality figures for docs/book/transcranial_ultrasound.md.
All physics computed by kwavers (Rust); this file contains only matplotlib
rendering.  Requires pykwavers to be installed.

Output directory: docs/book/figures/ch16/

References
----------
Aubry et al. (2003) J. Acoust. Soc. Am. 113(1):84
Clement & Hynynen (2002) Phys. Med. Biol. 47(8):1219
Schneider et al. (1996) Phys. Med. Biol. 41:111
Marechal (1947) Rev. d'Optique 26:257
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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


def fig01_skull_insertion_loss() -> None:
    f_MHz = np.linspace(0.25, 2.0, 300)
    alpha0 = 6.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for L_mm, col, lbl in [
        (3, "#1f77b4", "3 mm (thin)"),
        (6, "#ff7f0e", "6 mm (typical)"),
        (9, "#2ca02c", "9 mm (thick)"),
    ]:
        IL = kw.skull_insertion_loss_two_way_db(f_MHz, L_mm * 1e-3, alpha0, 1.2)
        ax.plot(f_MHz, IL, color=col, label=lbl)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Two-way insertion loss (dB)")
    ax.set_title(r"Skull insertion loss: $\alpha(f) = \alpha_0 f^{1.2}$, two-way")
    ax.legend(title="Skull thickness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig01_skull_insertion_loss")
    plt.close(fig)


def fig02_phase_aberration() -> None:
    N = 128
    x = np.linspace(-30, 30, N)
    sigma_phi = 3.0

    ideal = np.ones(N)
    phase_screen = kw.skull_phase_screen(N, sigma_phi, 42)
    aberrated = np.exp(1j * phase_screen)
    corrected = np.ones(N, dtype=complex)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(x, np.real(ideal), color="#1f77b4")
    axes[0].set_ylabel("Re(wavefront)")
    axes[0].set_title("Ideal flat wavefront (no skull)")
    axes[0].set_ylim(-2, 2)

    axes[1].plot(x, np.real(aberrated), color="#d62728")
    axes[1].set_ylabel("Re(wavefront)")
    axes[1].set_title(f"After skull aberration (sigma_phi = {sigma_phi:.0f} rad)")
    axes[1].set_ylim(-2, 2)

    axes[2].plot(x, np.real(corrected), color="#2ca02c")
    axes[2].set_ylabel("Re(wavefront)")
    axes[2].set_title("After time-reversal phase conjugation (corrected)")
    axes[2].set_ylim(-2, 2)
    axes[2].set_xlabel("Transducer element position (mm)")

    fig.tight_layout()
    savefig("fig02_phase_aberration")
    plt.close(fig)


def fig03_ct_conversion() -> None:
    HU = np.linspace(-1000, 3000, 500)
    rho = kw.hu_to_density_schneider(HU)
    c = kw.hu_to_sound_speed_schneider(HU)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(HU, rho, color="#1f77b4")
    ax1.axvline(0, color="k", linewidth=0.5, linestyle=":")
    ax1.axvline(1000, color="gray", linewidth=0.5, linestyle="--", label="HU=1000 (cortical bone)")
    ax1.set_xlabel("CT number (HU)")
    ax1.set_ylabel(r"Density $\rho$ (g/cm3)")
    ax1.set_title("HU -> density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(HU, c, color="#d62728")
    ax2.axvline(0, color="k", linewidth=0.5, linestyle=":")
    ax2.axvline(1000, color="gray", linewidth=0.5, linestyle="--", label="HU=1000")
    ax2.axhline(1500, color="blue", linewidth=0.5, linestyle=":", label="c = 1500 m/s (water)")
    ax2.set_xlabel("CT number (HU)")
    ax2.set_ylabel("Sound speed $c$ (m/s)")
    ax2.set_title("HU -> speed of sound (Schneider 1996)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig03_ct_conversion")
    plt.close(fig)


def fig04_strehl_ratio() -> None:
    sigma_phi = np.linspace(0, 4.0, 300)
    S = kw.strehl_ratio(sigma_phi)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sigma_phi, S, color="#1f77b4")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="S = 0.5 (3 dB loss)")
    ax.axhline(0.2, color="orange", linestyle=":", linewidth=1, label="S = 0.2 (7 dB loss)")
    for s in [1.5, 2.5, 3.5]:
        ax.axvline(s, color="gray", linewidth=0.5, linestyle="-.")
    ax.set_xlabel(r"RMS phase aberration $\sigma_\phi$ (rad)")
    ax.set_ylabel(r"Strehl ratio $S = e^{-\sigma_\phi^2}$")
    ax.set_title("Transcranial focus degradation (Marechal approximation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    savefig("fig04_strehl_ratio")
    plt.close(fig)


def fig05_skull_temperature() -> None:
    t = np.linspace(0, 30, 300)
    alpha_skull = 40.0
    I_skull = 1e4
    rho_skull = 1900.0
    C_skull = 1300.0
    alpha_brain = 1.0
    I_brain = 500.0
    rho_brain = 1040.0
    C_brain = 3600.0

    dT_skull = kw.skull_surface_temperature_rise(alpha_skull, I_skull, t, rho_skull, C_skull)
    dT_brain = kw.skull_surface_temperature_rise(alpha_brain, I_brain, t, rho_brain, C_brain)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, dT_skull, color="#d62728", label="Skull surface (cortical bone)")
    ax.plot(t, dT_brain, color="#1f77b4", linestyle="--", label="Brain target (soft tissue)")
    ax.axhline(1.0, color="orange", linestyle=":", linewidth=1.5, label="dT = 1 degC (guidance)")
    ax.axhline(6.0, color="r", linestyle="--", linewidth=1.5, label="dT = 6 degC (damage threshold)")
    ax.set_xlabel("Exposure time $t$ (s)")
    ax.set_ylabel("Temperature rise dT (degC)")
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
