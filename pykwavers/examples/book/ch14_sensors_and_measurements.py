"""
Chapter 14 figure generation — Sensors and Measurements
========================================================

Produces publication-quality figures for docs/book/sensors_and_measurements.md.

Output directory: docs/book/figures/ch14/

Figures produced
----------------
fig01  Hydrophone directivity: sinc pattern H(θ) = sinc(kd sinθ / π) vs angle
fig02  Spatial Nyquist: grating lobe position vs d/λ ratio
fig03  Pressure-velocity relationship: |p|, |u| on-axis for a plane wave
fig04  Time-reversal focusing: schematic demonstration (analytical Green's fn)
fig05  Side-by-side: measured vs reconstructed PA time-of-flight signal

References
----------
Selfridge et al. (1980) IEEE Trans SU-27
IEC 61685 (2001) Ultrasonic measurement standards
Fink (1992) IEEE Trans UFFC 39(5):555
Xu & Wang (2006) Rev. Mod. Phys. 78:1338
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch14")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch14/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

C0 = 1500.0


# ── Figure 01: Hydrophone directivity ────────────────────────────────────────
def fig01_hydrophone_directivity() -> None:
    """
    Rectangular active area a×b: H(θ_x, θ_y) = sinc(ka sinθ_x)·sinc(kb sinθ_y)
    For square element a=b, azimuth cut: H(θ) = sinc(ka sinθ / π).
    """
    theta = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 3600)
    f = 5.0e6   # 5 MHz
    k = 2 * np.pi * f / C0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for d_um, col in [(100, "#1f77b4"), (200, "#ff7f0e"), (500, "#2ca02c"), (1000, "#d62728")]:
        d = d_um * 1e-6
        ka_sin = k * d * np.sin(theta)
        H = np.sinc(ka_sin / np.pi)   # numpy sinc already normalised
        H_dB = 20 * np.log10(np.abs(H) + 1e-12)
        axes[0].plot(np.degrees(theta), H**2, color=col, label=f"$d={d_um}$ µm")
        axes[1].plot(np.degrees(theta), np.clip(H_dB, -60, 0), color=col, label=f"$d={d_um}$ µm")

    for ax in axes:
        ax.set_xlabel(r"Angle $\theta$ (°)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel(r"$|H(\theta)|^2$")
    axes[0].set_title("Hydrophone directivity (linear)")
    axes[1].set_ylabel(r"$|H(\theta)|$ (dB)")
    axes[1].set_title("Hydrophone directivity (dB)")
    axes[1].set_ylim(-60, 5)
    fig.suptitle(r"Hydrophone directivity: $H(\theta)=\mathrm{sinc}(kd\sin\theta/\pi)$, $f=5$ MHz", y=1.01)
    fig.tight_layout()
    savefig("fig01_hydrophone_directivity")
    plt.close(fig)


# ── Figure 02: Spatial Nyquist and grating lobes ─────────────────────────────
def fig02_grating_lobes() -> None:
    """
    Grating lobe appears at θ_g = arcsin(n λ/d) for n = ±1, ±2, ...
    Main lobe at θ_s: grating lobes at sinθ_g = sinθ_s ± nλ/d.
    For steering at 0°: first grating lobe at sinθ_g = λ/d.
    Condition to avoid: d ≤ λ/2.
    """
    theta_s = 0.0   # steering angle
    theta = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 3600)
    f = 3.0e6
    lam = C0 / f
    N = 32

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for d_lambda, col, lbl in [(0.5, "#1f77b4", r"$d=\lambda/2$ (Nyquist)"),
                                (1.0, "#ff7f0e", r"$d=\lambda$ (grating lobe at ±90°)"),
                                (1.5, "#2ca02c", r"$d=1.5\lambda$ (grating lobe at 42°)")]:
        d = d_lambda * lam
        k = 2 * np.pi / lam
        phase = k * d * (np.sin(theta)[:, None] - np.sin(theta_s)) * np.arange(N)[None, :]
        AF = np.abs(np.sum(np.exp(1j * phase), axis=1)) / N
        AF_dB = 20 * np.log10(AF + 1e-12)
        axes[0].plot(np.degrees(theta), AF**2, color=col, label=lbl)
        axes[1].plot(np.degrees(theta), np.clip(AF_dB, -60, 0), color=col, label=lbl)

    for ax in axes:
        ax.set_xlabel(r"Angle $\theta$ (°)")
        ax.legend(fontsize=8)
        ax.axvline(0, color="k", linewidth=0.5)

    axes[0].set_ylabel(r"Intensity $|AF|^2$")
    axes[0].set_title("Grating lobes (linear scale)")
    axes[1].set_ylabel(r"$|AF|$ (dB)")
    axes[1].set_title("Grating lobes (dB scale)")
    axes[1].set_ylim(-60, 5)

    fig.suptitle(f"Spatial aliasing: $N={N}$, $f={f*1e-6:.0f}$ MHz", y=1.01)
    fig.tight_layout()
    savefig("fig02_grating_lobes")
    plt.close(fig)


# ── Figure 03: Pressure-velocity plane wave relationship ─────────────────────
def fig03_pressure_velocity() -> None:
    """
    For a plane wave: p = ρ c u  (specific acoustic impedance Z = ρc).
    In time domain: p(x,t) = P0 sin(kx - ωt)
                    u(x,t) = P0/(ρc) sin(kx - ωt)  (co-phase)
    Show both at fixed time, plus impedance as function of medium.
    """
    rho = 998.0
    c = C0
    Z = rho * c
    P0 = 1e5  # Pa
    f = 1e6
    k = 2 * np.pi * f / c
    t = 0.0

    x = np.linspace(0, 3e-3, 500)
    p = P0 * np.sin(k * x - 2 * np.pi * f * t)
    u = P0 / Z * np.sin(k * x - 2 * np.pi * f * t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1_u = ax1.twinx()
    l1, = ax1.plot(x * 1e3, p * 1e-3, color="#1f77b4", label="$p(x)$ (kPa)")
    l2, = ax1_u.plot(x * 1e3, u * 100, color="#d62728", linestyle="--", label="$u(x)$ (cm/s)")
    ax1.set_xlabel("Position $x$ (mm)")
    ax1.set_ylabel("Pressure $p$ (kPa)", color="#1f77b4")
    ax1_u.set_ylabel("Particle velocity $u$ (cm/s)", color="#d62728")
    ax1.set_title(r"Plane wave: $p = \rho c u$")
    ax1.legend(handles=[l1, l2])

    # Impedance by medium
    media = [("Air", 1.2, 343), ("Water", 998, 1481), ("Liver", 1060, 1560),
             ("Bone", 1900, 3500), ("Steel", 7800, 5900)]
    names = [m[0] for m in media]
    Z_vals = [m[1] * m[2] / 1e6 for m in media]  # MRayl
    ax2.bar(names, Z_vals, color=plt.cm.tab10(np.linspace(0, 0.5, len(media))))
    ax2.set_ylabel("Acoustic impedance $Z = \\rho c$ (MRayl)")
    ax2.set_title("Impedance $Z$ for common media")
    ax2.set_yscale("log")

    fig.tight_layout()
    savefig("fig03_pressure_velocity")
    plt.close(fig)


# ── Figure 04: Time-reversal focusing envelope ───────────────────────────────
def fig04_time_reversal() -> None:
    """
    Illustrate TR: a point source at origin emits, array records.
    Time-reversed signal retransmitted focuses back at origin.
    Show spatial focus quality (array factor) for N elements.
    """
    N_arr = [4, 16, 64]
    f = 1e6
    lam = C0 / f
    d = lam / 2  # half-wavelength spacing

    theta = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 3600)
    k = 2 * np.pi / lam

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for N, col in zip(N_arr, colors):
        phase = k * d * np.sin(theta)[:, None] * np.arange(N)[None, :]
        AF = np.abs(np.sum(np.exp(1j * phase), axis=1)) / N
        AF_dB = 20 * np.log10(AF + 1e-12)
        ax.plot(np.degrees(theta), np.clip(AF_dB, -60, 0), color=col, label=f"$N={N}$")

    ax.set_xlabel(r"Angle $\theta$ (°)")
    ax.set_ylabel("Normalised focus quality (dB)")
    ax.set_title("Time-reversal focus quality vs array size\n"
                 r"$d = \lambda/2$, coherent gain = 20$\log_{10}N$ dB")
    ax.legend()
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig04_time_reversal")
    plt.close(fig)


# ── Figure 05: Sensor recording: ideal vs noisy signal (side-by-side) ─────────
def fig05_signal_comparison() -> None:
    """
    Analytical N-wave photoacoustic signal (reference) vs noisy sensor recording.
    Demonstrates SNR improvement by averaging.
    """
    R = 0.001   # 1 mm sphere
    r_d = 0.04  # 40 mm
    c = C0
    t1 = (r_d - R) / c
    t2 = (r_d + R) / c
    t = np.linspace(0, 60e-6, 2000)
    t_c = (t1 + t2) / 2

    # Ideal signal
    p_ideal = np.zeros_like(t)
    mask = (t >= t1) & (t <= t2)
    p_ideal[mask] = (t[mask] - t_c) / ((t2 - t1) / 2)

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(t)) * 0.3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(t * 1e6, p_ideal + noise, color="#d62728", linewidth=0.8, label="Single measurement")
    axes[0].plot(t * 1e6, p_ideal, color="#1f77b4", linewidth=2, label="Analytical signal")
    axes[0].set_xlabel(r"Time (µs)")
    axes[0].set_ylabel("Normalised pressure")
    axes[0].set_title("Single PA measurement (SNR ≈ 3 dB)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Average of 64 realisations
    averaged = p_ideal.copy()
    for _ in range(64):
        averaged = averaged + rng.standard_normal(len(t)) * 0.3
    averaged /= 65.0

    axes[1].plot(t * 1e6, averaged, color="#d62728", linewidth=0.8, label="64-shot average")
    axes[1].plot(t * 1e6, p_ideal, color="#1f77b4", linewidth=2, label="Analytical signal")
    axes[1].set_xlabel(r"Time (µs)")
    axes[1].set_ylabel("Normalised pressure")
    axes[1].set_title("64-shot average (SNR ≈ 21 dB)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Sensor noise averaging: SNR improvement by $\\sqrt{N}$", y=1.01)
    fig.tight_layout()
    savefig("fig05_signal_comparison")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 14 figures (Sensors and Measurements)...")
    fig01_hydrophone_directivity()
    fig02_grating_lobes()
    fig03_pressure_velocity()
    fig04_time_reversal()
    fig05_signal_comparison()
    print("Done. Output: docs/book/figures/ch14/")
