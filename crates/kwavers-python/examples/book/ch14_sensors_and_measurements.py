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

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
    Rectangular active area a×b: H(θ) = sinc(kd sinθ / π)  (azimuth cut, square element).
    For circular piston: H(θ) = 2 J₁(ka sinθ) / (ka sinθ) — available via
    kw.circular_piston_directivity but is a different aperture geometry.
    No rectangular-aperture sinc binding exists; sinc is closed-form and
    computed directly from numpy (not physics, pure geometry).
    """
    theta = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 3600)
    f = 5.0e6   # 5 MHz
    k = 2 * np.pi * f / C0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for d_um, col in [(100, "#1f77b4"), (200, "#ff7f0e"), (500, "#2ca02c"), (1000, "#d62728")]:
        d = d_um * 1e-6
        ka_sin = k * d * np.sin(theta)
        H = np.sinc(ka_sin / np.pi)   # numpy sinc: sinc(x) = sin(πx)/(πx)
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
    Array factor computed via kw.linear_array_factor(theta, k, d, N, steer_rad).
    Condition to avoid grating lobes: d ≤ λ/2.
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig02 (grating lobes, linear_array_factor)")
    theta_s = 0.0   # steering angle
    theta = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 3600)
    f = 3.0e6
    lam = C0 / f
    k = 2 * np.pi / lam
    N = 32

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for d_lambda, col, lbl in [(0.5, "#1f77b4", r"$d=\lambda/2$ (Nyquist)"),
                                (1.0, "#ff7f0e", r"$d=\lambda$ (grating lobe at ±90°)"),
                                (1.5, "#2ca02c", r"$d=1.5\lambda$ (grating lobe at 42°)")]:
        d = d_lambda * lam
        AF = np.asarray(kw.linear_array_factor(theta, k, d, N, theta_s))
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
    Plane wave: p = ρ c u  (specific acoustic impedance Z = ρc).
    p(x,t) = P₀ sin(kx − ωt),  u(x,t) = P₀/(ρc) sin(kx − ωt).
    Impedance bar: Z = ρc from kw.tissue_properties for Water and Liver;
    Air (Duck 1990) and Cortical bone (Duck 1990) and Steel kept as reference.
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig03 (tissue impedance via tissue_properties)")
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

    # Impedance by medium: bindings used for tissues with kw.tissue_properties
    def _z_mrayl(key: str) -> float:
        _c, _rho, *_ = kw.tissue_properties(key)
        return _c * _rho / 1e6

    media = [
        ("Air",    0.000413),            # Duck (1990) — no kw binding for gas
        ("Water",  _z_mrayl("water")),   # kw.tissue_properties
        ("Liver",  _z_mrayl("liver")),   # kw.tissue_properties
        ("Bone",   7.38),                # Duck (1990) cortical bone — no kw binding
        ("Steel",  46.1),                # reference value — no kw binding
    ]
    names = [m[0] for m in media]
    Z_vals = [m[1] for m in media]  # MRayl
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
    TR focus quality: coherent array factor for steering at 0° with d = λ/2.
    Computed via kw.linear_array_factor(theta, k, d, N, steer_rad=0.0).
    Coherent gain at broadside = 20 log₁₀(N) dB.
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig04 (TR focus, linear_array_factor)")
    N_arr = [4, 16, 64]
    f = 1e6
    lam = C0 / f
    d = lam / 2  # half-wavelength spacing
    theta = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, 3600)
    k = 2 * np.pi / lam

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for N, col in zip(N_arr, colors):
        AF = np.asarray(kw.linear_array_factor(theta, k, d, N, 0.0))
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
    PA sphere N-wave (reference) vs noisy sensor recording.
    Ideal signal via kw.pa_sphere_pressure_signal (Xu & Wang 2006).
    Demonstrates SNR improvement by signal averaging (√N law).
    """
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for fig05 (PA sphere signal)")
    R = 0.001    # 1 mm sphere
    r_d = 0.04   # 40 mm detector
    c = C0
    Gamma = 0.2
    mu_a = 100.0    # m⁻¹
    Phi = 1.0       # J/m² — initial pressure p₀ = Γ μ_a Φ
    p0 = Gamma * mu_a * Phi   # 20 Pa

    t = np.linspace(0, 60e-6, 2000)
    # Analytical N-wave via Rust kernel (Xu & Wang 2006)
    p_ideal = np.asarray(kw.pa_sphere_pressure_signal(t, R, Gamma, mu_a, c, r_d, p0))
    # Normalise to [-1, 1] for comparison with noisy signal
    p_scale = np.abs(p_ideal).max() + 1e-30
    p_ideal = p_ideal / p_scale

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
