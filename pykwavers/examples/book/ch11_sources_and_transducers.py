"""
Chapter 11 figure generation — Sources and Transducers
=======================================================

Produces publication-quality figures for docs/book/sources_and_transducers.md.
All expressions from classical transducer theory (piston source, focused bowl,
phased array) are closed-form analytical solutions.

Output directory: docs/book/figures/ch11/

Figures produced
----------------
fig01  Piston source directivity H(θ) = 2J₁(ka sinθ)/(ka sinθ) vs angle
fig02  On-axis pressure of a focused bowl transducer vs depth
fig03  Linear phased array beam pattern: steering at 0°, 15°, 30°
fig04  Delay law for a linear array: element delays vs steering angle
fig05  BLI rasterization accuracy vs grid points per wavelength

References
----------
O'Neil (1949) J. Acoust. Soc. Am. 21:516
Thomenius (1996) Proc. IEEE Ultrasonics Symposium
Selfridge et al. (1980) IEEE Trans. SU-27:19
"""

from __future__ import annotations

import os
import numpy as np
from scipy.special import j1
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch11")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch11/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

C0 = 1500.0   # m/s


# ── Figure 01: Piston directivity ─────────────────────────────────────────────
def fig01_piston_directivity() -> None:
    """
    H(θ) = 2 J₁(ka sinθ) / (ka sinθ)
    Sinc-like pattern; first null at ka sinθ = 1.22π.
    """
    theta = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 3600)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ka_values = [(2.0, "#1f77b4"), (5.0, "#ff7f0e"), (10.0, "#2ca02c"), (20.0, "#d62728")]

    for ka, col in ka_values:
        x = ka * np.sin(theta)
        H = np.where(np.abs(x) < 1e-10, 1.0, 2 * j1(x) / x)
        H_dB = 20 * np.log10(np.abs(H) + 1e-12)
        axes[0].plot(np.degrees(theta), H**2, color=col, label=f"$ka = {ka:.0f}$")
        axes[1].plot(np.degrees(theta), np.clip(H_dB, -60, 0), color=col, label=f"$ka = {ka:.0f}$")

    for ax in axes:
        ax.set_xlabel(r"Angle $\theta$ (°)")
        ax.legend(fontsize=8)
        ax.axvline(0, color="k", linewidth=0.5)

    axes[0].set_ylabel(r"Intensity pattern $|H(\theta)|^2$")
    axes[0].set_title("Piston directivity (linear scale)")
    axes[1].set_ylabel(r"$|H(\theta)|$ (dB)")
    axes[1].set_title("Piston directivity (dB scale)")
    axes[1].set_ylim(-60, 5)

    fig.suptitle(r"Piston directivity: $H(\theta) = 2J_1(ka\sin\theta)/(ka\sin\theta)$", y=1.01)
    fig.tight_layout()
    savefig("fig01_piston_directivity")
    plt.close(fig)


# ── Figure 02: Focused bowl on-axis pressure ──────────────────────────────────
def fig02_focused_bowl_onaxis() -> None:
    """
    On-axis pressure of a focused spherical bowl transducer (O'Neil 1949):
      p(z) = 2ρcU₀ sin(k/2 · |z·(1/cos α_z - 1)|) · exp(ikz)
    where z is depth along axis, F is focal length, a is aperture radius,
    and α_z = arctan(a/z).
    Simplified (Zemanek 1971 paraxial approximation):
      |p(z)|/|p_focus| ≈ |sin((ka²/8F)·(z-F)/F·(1+z/F))| — near focus
    Use exact Rayleigh integral result on axis:
      p(z) = ρcU₀(exp(ikz) - exp(ik√(z²+a²))) / (1 - cos α)
    """
    F = 0.06    # 60 mm focal length
    a = 0.015   # 15 mm aperture radius
    f = 1.0e6   # 1 MHz
    k = 2 * np.pi * f / C0
    rho = 998.0

    z = np.linspace(0.001, 0.12, 3000)  # 1 mm to 120 mm
    r1 = np.sqrt(z**2 + a**2)           # distance from bowl rim to on-axis point

    # Exact on-axis pressure integral (normalised):
    # p(z) = rho c U0 (e^{ikz} - e^{ikr1})
    # |p| = rho c U0 * 2 |sin(k(r1-z)/2)|
    p_norm = 2 * np.abs(np.sin(k * (r1 - z) / 2))
    p_focus = 2 * np.abs(np.sin(k * (np.sqrt(F**2 + a**2) - F) / 2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z * 1e3, p_norm / p_focus, color="#1f77b4")
    ax.axvline(F * 1e3, color="k", linestyle="--", linewidth=1, label=f"Focus z=F={F*1000:.0f} mm")
    ax.set_xlabel("Axial depth $z$ (mm)")
    ax.set_ylabel("Normalised pressure $|p|/|p_F|$")
    ax.set_title(f"Focused bowl on-axis pressure\n"
                 f"$F={F*1e3:.0f}$ mm, $a={a*1e3:.0f}$ mm, $f={f*1e-6:.0f}$ MHz, $c={C0:.0f}$ m/s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig02_focused_bowl_onaxis")
    plt.close(fig)


# ── Figure 03: Linear phased array beam pattern ───────────────────────────────
def fig03_array_beam_pattern() -> None:
    """
    Far-field array factor: AF(θ) = Σ_{n=0}^{N-1} w_n exp(i·n·kd·(sinθ - sinθ_s))
    Rectangular aperture: w_n = 1.
    """
    N = 64          # elements
    d = 0.3e-3      # 0.3 mm pitch (half-wavelength at 2.5 MHz)
    f = 2.5e6
    k = 2 * np.pi * f / C0

    theta = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 3600)
    steer_angles_deg = [0.0, 15.0, 30.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for ang_deg, col in zip(steer_angles_deg, colors):
        theta_s = np.radians(ang_deg)
        # Array factor (rectangular apodization)
        phase = k * d * (np.sin(theta)[:, None] - np.sin(theta_s)) * np.arange(N)[None, :]
        AF = np.abs(np.sum(np.exp(1j * phase), axis=1)) / N
        AF_dB = 20 * np.log10(AF + 1e-12)
        ax.plot(np.degrees(theta), np.clip(AF_dB, -60, 0),
                color=col, label=rf"$\theta_s = {ang_deg:.0f}°$")

    ax.set_xlabel(r"Angle $\theta$ (°)")
    ax.set_ylabel("Normalised array factor (dB)")
    ax.set_title(f"Linear phased array beam pattern\n$N={N}$, $d={d*1e3:.1f}$ mm, $f={f*1e-6:.1f}$ MHz")
    ax.set_ylim(-60, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig03_array_beam_pattern")
    plt.close(fig)


# ── Figure 04: Delay law ──────────────────────────────────────────────────────
def fig04_delay_law() -> None:
    """
    tau_i = (|r_i - r_f| - min_j |r_j - r_f|) / c
    For linear array steered to angle theta_s or focused to (x_f, z_f).
    """
    N = 64
    d = 0.3e-3
    elem_x = (np.arange(N) - (N - 1) / 2) * d

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Steering only (far-field)
    for ang_deg, col, ls in [(0.0, "#1f77b4", "solid"), (15.0, "#ff7f0e", "dashed"), (30.0, "#2ca02c", "dotted")]:
        theta_s = np.radians(ang_deg)
        tau = elem_x * np.sin(theta_s) / C0
        tau -= tau.min()
        axes[0].plot(np.arange(N), tau * 1e6, color=col, linestyle=ls, label=rf"$\theta_s={ang_deg:.0f}°$")

    axes[0].set_xlabel("Element index")
    axes[0].set_ylabel(r"Delay $\tau_i$ (µs)")
    axes[0].set_title("Steering delay law (linear)")
    axes[0].legend()

    # Focused (near-field)
    F = 0.05  # 50 mm focus depth
    for x_f, col, lbl in [(0.0, "#1f77b4", "F=(0, 50mm)"),
                           (0.01, "#ff7f0e", "F=(10, 50mm)"),
                           (0.02, "#2ca02c", "F=(20, 50mm)")]:
        r_f = np.array([x_f, 0.0, F])
        r_i = np.stack([elem_x, np.zeros(N), np.zeros(N)], axis=1)
        dist = np.linalg.norm(r_i - r_f, axis=1)
        tau = (dist - dist.min()) / C0
        axes[1].plot(np.arange(N), tau * 1e6, label=lbl)

    axes[1].set_xlabel("Element index")
    axes[1].set_ylabel(r"Delay $\tau_i$ (µs)")
    axes[1].set_title("Focusing delay law (near-field)")
    axes[1].legend()

    fig.suptitle(r"Delay law: $\tau_i = (|\mathbf{r}_i - \mathbf{r}_f| - \min_j|\mathbf{r}_j - \mathbf{r}_f|)/c$", y=1.01)
    fig.tight_layout()
    savefig("fig04_delay_law")
    plt.close(fig)


# ── Figure 05: BLI rasterization accuracy ─────────────────────────────────────
def fig05_bli_accuracy() -> None:
    """
    Rasterization error vs points per wavelength (PPW).
    For a sinusoidal source mask, BLI preserves the exact continuous
    Fourier content up to the Nyquist limit. Error decays as 1/PPW.
    Analytical: aliasing error ~ sinc²(π/PPW) for spectral content at Nyquist.
    """
    ppw = np.linspace(2.1, 20.0, 200)  # points per wavelength

    # Spectral leakage from sampling (normalised Fourier amplitude at Nyquist)
    # For a rect window of N points: worst-case aliasing ~ 1/N
    aliasing_error_dB = 20 * np.log10(1.0 / ppw)  # linear decay in amplitude

    # More accurate BLI bound: residual after Whittaker-Shannon interpolation
    # E(ppw) = max_x |f(x) - f_BLI(x)| ~ π²/(6 ppw²) for band-limited signals
    bli_error_dB = 20 * np.log10(np.pi**2 / (6 * ppw**2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ppw, aliasing_error_dB, label="Nearest-neighbour rasterization error")
    ax.plot(ppw, bli_error_dB, "--", label="BLI rasterization error (Whittaker-Shannon)")
    ax.axvline(4, color="gray", linestyle=":", linewidth=1, label="PPW = 4 (typical k-Wave default)")
    ax.set_xlabel("Points per wavelength (PPW)")
    ax.set_ylabel("Amplitude error (dB)")
    ax.set_title("Rasterization accuracy vs grid resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-80, 5)
    fig.tight_layout()
    savefig("fig05_bli_accuracy")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 11 figures (Sources and Transducers)...")
    fig01_piston_directivity()
    fig02_focused_bowl_onaxis()
    fig03_array_beam_pattern()
    fig04_delay_law()
    fig05_bli_accuracy()
    print("Done. Output: docs/book/figures/ch11/")
