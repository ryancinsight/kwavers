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
fig05  BLI interpolation accuracy from the Rust BLI stencil kernel
fig06  Acoustic lens delay law and Fresnel zone radii
fig07  Isoplanatic corrective-lens steering

References
----------
O'Neil (1949) J. Acoust. Soc. Am. 21:516
Thomenius (1996) Proc. IEEE Ultrasonics Symposium
Selfridge et al. (1980) IEEE Trans. SU-27:19
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
    H(θ) = 2 J₁(ka sinθ) / (ka sinθ)  — O'Neil (1949).
    Computed via kw.circular_piston_directivity (Rust kernel, normalised to unity on-axis).
    Sinc-like pattern; first null at ka sinθ = 1.22π.
    """
    theta = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 3600)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ka_values = [(2.0, "#1f77b4"), (5.0, "#ff7f0e"), (10.0, "#2ca02c"), (20.0, "#d62728")]

    for ka, col in ka_values:
        H = np.asarray(kw.circular_piston_directivity(theta, ka))
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
    On-axis pressure of a focused spherical bowl transducer (O'Neil 1949).
    Computed via kw.focused_bowl_onaxis (Rust kernel, exact Rayleigh integral on axis).
    Args: bowl_radius_m = aperture radius a, focal_length_m = F.
    """
    F = 0.06    # 60 mm focal length
    a = 0.015   # 15 mm aperture radius
    f = 1.0e6   # 1 MHz

    z = np.linspace(0.001, 0.12, 3000)  # 1 mm to 120 mm
    # On-axis pressure [Pa] with unit source amplitude (p0_pa=1.0)
    p_pa = np.asarray(kw.focused_bowl_onaxis(z, a, F, f, 1.0, C0))
    # Normalise by pressure at the geometric focus
    p_focus_val = float(np.asarray(kw.focused_bowl_onaxis(np.array([F]), a, F, f, 1.0, C0))[0])
    p_norm = p_pa / (p_focus_val if p_focus_val > 0.0 else 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z * 1e3, p_norm, color="#1f77b4")
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
    Far-field array factor: AF(θ) = Σ_{n=0}^{N-1} exp(i·n·kd·(sinθ - sinθ_s)).
    Computed via kw.linear_array_factor (Rust kernel, rectangular apodization).
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
        AF = np.asarray(kw.linear_array_factor(theta, k, d, N, theta_s))
        AF_dB = 20 * np.log10(np.abs(AF) + 1e-12)
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
    tau_i = (|r_i - r_f| - min_j |r_j - r_f|) / c.
    Computed via kw.delay_law_focus_2d (Rust kernel).
    Steering: focus placed at (z_far · sin θ_s, z_far) with z_far = 1e3 m (far-field limit).
    Focusing: near-field focus at (x_f, z_f) in the (x, z) plane.
    """
    N = 64
    d = 0.3e-3
    elem_x = (np.arange(N) - (N - 1) / 2) * d
    elem_z_zeros = np.zeros(N)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Steering (far-field): focus at 1 km in direction θ_s
    Z_FAR = 1.0e3
    for ang_deg, col, ls in [(0.0, "#1f77b4", "solid"), (15.0, "#ff7f0e", "dashed"), (30.0, "#2ca02c", "dotted")]:
        theta_s = np.radians(ang_deg)
        x_far = Z_FAR * np.sin(theta_s)
        tau = np.asarray(kw.delay_law_focus_2d(elem_x, elem_z_zeros, x_far, Z_FAR, C0))
        axes[0].plot(np.arange(N), tau * 1e6, color=col, linestyle=ls, label=rf"$\theta_s={ang_deg:.0f}°$")

    axes[0].set_xlabel("Element index")
    axes[0].set_ylabel(r"Delay $\tau_i$ (µs)")
    axes[0].set_title("Steering delay law (far-field focus)")
    axes[0].legend()

    # Focused (near-field)
    F = 0.05  # 50 mm focus depth
    for x_f, col, lbl in [(0.0, "#1f77b4", "F=(0, 50mm)"),
                           (0.01, "#ff7f0e", "F=(10, 50mm)"),
                           (0.02, "#2ca02c", "F=(20, 50mm)")]:
        tau = np.asarray(kw.delay_law_focus_2d(elem_x, elem_z_zeros, x_f, F, C0))
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
    Sub-sample interpolation error vs points per wavelength (PPW).

    Rust computes both the BLI stencil weights and the deterministic sinusoid
    reconstruction error curves; Python only converts RMS error to dB and plots.
    """
    ppw = np.linspace(2.25, 20.0, 160)  # points per wavelength
    delta = np.linspace(0.0, 1.0, 256, endpoint=False)
    n_stencil = 8
    nn_rms, bli_rms = kw.bli_interpolation_error_curves(
        np.ascontiguousarray(ppw),
        np.ascontiguousarray(delta),
        n_stencil,
    )

    nn_error_dB = 20 * np.log10(np.maximum(nn_rms, 1e-15))
    bli_error_dB = 20 * np.log10(np.maximum(bli_rms, 1e-15))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ppw, nn_error_dB, label="Nearest-neighbour sub-sample error")
    ax.plot(ppw, bli_error_dB, "--", label=f"BLI stencil error (N={n_stencil})")
    ax.axvline(4, color="gray", linestyle=":", linewidth=1, label="PPW = 4 (typical k-Wave default)")
    ax.set_xlabel("Points per wavelength (PPW)")
    ax.set_ylabel("RMS interpolation error (dB re unit amplitude)")
    ax.set_title("Rust BLI stencil accuracy vs grid resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-80, 5)
    fig.tight_layout()
    savefig("fig05_bli_accuracy")
    plt.close(fig)


# ── Figure 06: Acoustic lenses (static refractive + Fresnel zone plate) ───────
def fig06_acoustic_lens() -> None:
    """
    (a) The static refractive lens imposes the same focusing delay as the
        phased-array delay law: tau(r)=(sqrt(F^2+r^2)-F)/c
        (kw.acoustic_lens_delay_profile), matching the paraxial r^2/(2cF) for
        r << F -- a lens IS a passive delay law.
    (b) The Fresnel zone-plate boundary radii r_n=sqrt(n*lam*F+(n*lam/2)^2)
        (kw.fresnel_zone_radii) bunch as sqrt(n); alternate zones focus by
        diffraction.
    """
    F = 0.05            # 50 mm focal length
    aperture = 0.030    # 30 mm full aperture
    radii = np.linspace(0.0, aperture / 2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Static lens focusing delay tau(r) -- the lens IS a passive delay law.
    #     All physics from kw.acoustic_lens_delay_profile (Rust); Python only plots.
    tau = np.asarray(kw.acoustic_lens_delay_profile(radii, F, aperture, C0))
    axes[0].plot(radii * 1e3, tau * 1e6, lw=2, label=r"lens $\tau(r)=(\sqrt{F^2+r^2}-F)/c$")
    axes[0].set_xlabel("Aperture radius r (mm)")
    axes[0].set_ylabel(r"Focusing delay $\tau$ (µs)")
    axes[0].set_title("Static lens = passive delay law (F = 50 mm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) Fresnel zone-plate boundary radii (kw.fresnel_zone_radii, Rust) at two
    #     frequencies; the sqrt(n) bunching is visible directly in the markers.
    for freq_mhz, ax_color in [(1.0, "#1f77b4"), (3.0, "#d62728")]:
        wavelength_m = C0 / (freq_mhz * 1e6)  # input argument to the binding
        zr = np.asarray(kw.fresnel_zone_radii(F, wavelength_m, aperture / 2))
        n = np.arange(1, len(zr) + 1)
        axes[1].plot(n, zr * 1e3, "o-", color=ax_color, label=f"{freq_mhz:.0f} MHz ({len(zr)} zones)")
    axes[1].set_xlabel("Zone index n")
    axes[1].set_ylabel(r"Zone radius $r_n$ (mm)")
    axes[1].set_title(r"Fresnel zone plate: $r_n=\sqrt{n\lambda F+(n\lambda/2)^2}$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Acoustic lenses: refractive delay law (a) and diffractive zone plate (b)", y=1.01)
    fig.tight_layout()
    savefig("fig06_acoustic_lens")
    plt.close(fig)


# ── Figure 07: single-element corrective lens steering (Maimbourg 2020) ────────
def fig07_lens_steering() -> None:
    """
    Isoplanatic mechanical steering of a fixed corrective lens (Maimbourg 2020):
    theta_y=asin(x/F), T_z=F-sqrt(F^2-x^2) for F=61 mm, computed by
    kw.isoplanatic_steering_curve (Rust). The paper's Figure-2 table points are
    overlaid (literature data) and the +/-11 mm operating range is shaded.
    All physics is from kwavers; Python only plots and converts units.
    """
    F = 0.061  # 61 mm focal length (H101 transducer)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Steering geometry over ±15 mm; physical limit at |x|=F (kwavers kernel).
    x = np.linspace(-0.015, 0.015, 200)
    theta, t_z = kw.isoplanatic_steering_curve(x, F)
    theta = np.degrees(np.asarray(theta))  # rad -> deg (display)
    t_z = np.asarray(t_z) * 1e3            # m -> mm (display)
    ln1 = ax.plot(x * 1e3, theta, color="#1f77b4", label=r"$\theta_y=\arcsin(x/F)$")
    ax.set_xlabel("Transverse focus offset x (mm)")
    ax.set_ylabel(r"Rotation $\theta_y$ (deg)", color="#1f77b4")
    ax2 = ax.twinx()
    ln2 = ax2.plot(x * 1e3, t_z, color="#d62728", ls="--", label=r"$T_z=F-\sqrt{F^2-x^2}$")
    ax2.set_ylabel(r"Pullback $T_z$ (mm)", color="#d62728")
    # Maimbourg Fig. 2 table (literature data): x (mm) -> theta_y (deg + arcmin).
    tbl_x = np.array([2.3, 4.5, 6.8, 9.0, 11.2])
    tbl_theta = np.array([2 + 7/60, 4 + 14/60, 6 + 21/60, 8 + 28/60, 10 + 35/60])
    ln3 = ax.plot(tbl_x, tbl_theta, "o", color="#1f77b4", ms=5, label="Maimbourg 2020 table")
    ax.axvspan(-11, 11, color="gray", alpha=0.12, label="±11 mm operating range")
    ax.set_title("Isoplanatic mechanical steering of a corrective lens (F = 61 mm)")
    lns = ln1 + ln2 + ln3 + [ax.patches[-1]]
    ax.legend(lns, [l.get_label() for l in lns], loc="upper left", fontsize=8)

    fig.tight_layout()
    savefig("fig07_lens_steering")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 11 figures (Sources and Transducers)...")
    fig01_piston_directivity()
    fig02_focused_bowl_onaxis()
    fig03_array_beam_pattern()
    fig04_delay_law()
    fig05_bli_accuracy()
    fig06_acoustic_lens()
    fig07_lens_steering()
    print("Done. Output: docs/book/figures/ch11/")
