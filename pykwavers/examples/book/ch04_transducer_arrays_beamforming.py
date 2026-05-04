"""
Chapter 4: Transducer Arrays and Beamforming — Figure Generation Script
========================================================================

Generates all publication-quality figures for Chapter 4 of the kwavers book
using closed-form analytical expressions from:

  - O'Neil (1949) baffled piston directivity
  - Array factor theory (Szabo 2014)
  - Grating lobe positions (Theorem 4.2)
  - DAS resolution formulas (Theorems 4.4-4.5)
  - Apodization windows (Harris 1978)

Output directory: docs/book/figures/ch04/

Figures produced:
  fig01: Baffled piston element directivity D(θ) for several ka values
  fig02: Array factor and grating lobes for a 64-element array
  fig03: Apodization windows and their frequency response (sidelobe comparison)
  fig04: Lateral resolution vs depth for several apertures and f-numbers
  fig05: 2-D beam pattern (DAS, element × array factor product)
  fig06: BLI stencil weights vs off-grid offset

Usage::

    python ch04_transducer_arrays_beamforming.py

Requires: numpy, matplotlib, scipy
"""

import os
import numpy as np
from scipy.special import j1 as bessel_j1
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# ── Output directory ──────────────────────────────────────────────────────────

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch04")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch04/{name}.{{pdf,png}}")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
    }
)

C0 = 1540.0       # m/s (tissue average)
F0 = 5.0e6        # Hz center frequency
LAM = C0 / F0     # wavelength  ≈ 0.308 mm


# ─────────────────────────────────────────────────────────────────────────────
# Helper: baffled piston directivity D(θ) = 2 J₁(ka sinθ) / (ka sinθ)
# ─────────────────────────────────────────────────────────────────────────────


def piston_directivity(theta_deg: np.ndarray, ka: float) -> np.ndarray:
    """Baffled piston directivity |D(θ)| (Eq. 4.3), dB normalised to 0 dB on axis."""
    x = ka * np.sin(np.deg2rad(theta_deg))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.abs(x) < 1e-12, 1.0, 2.0 * bessel_j1(x) / x)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: Element directivity for several ka values
# ─────────────────────────────────────────────────────────────────────────────

print("[fig01] Baffled piston element directivity")

theta = np.linspace(-90, 90, 1800)
ka_vals = [1.0, 2.0, 4.0, 8.0]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(ka_vals)))

fig, ax = plt.subplots(figsize=(7, 4.5))
for ka, col in zip(ka_vals, colors):
    d = piston_directivity(theta, ka)
    d_dB = 20 * np.log10(np.abs(d) + 1e-12)
    ax.plot(theta, d_dB, color=col, label=f"$ka = {ka}$")

ax.axhline(-6, color="k", lw=0.8, ls=":", label="−6 dB")
ax.set_xlabel(r"Angle $\theta$ (°)")
ax.set_ylabel(r"$|D(\theta)|$ (dB)")
ax.set_title("Baffled Circular Piston Directivity (Eq. 4.3)")
ax.set_xlim(-90, 90)
ax.set_ylim(-60, 2)
ax.legend(ncol=2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.tight_layout()
savefig("fig01_element_directivity")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: Array factor and grating lobes — 64-element array
# ─────────────────────────────────────────────────────────────────────────────

print("[fig02] Array factor and grating lobes")

N_elements = 64
pitches_lambda = [0.5, 1.0, 2.0]  # d / λ
steer_deg = 20.0
steer_rad = np.deg2rad(steer_deg)
theta = np.linspace(-90, 90, 3600)
theta_rad = np.deg2rad(theta)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, d_ratio in zip(axes, pitches_lambda):
    d = d_ratio * LAM
    k = 2 * np.pi / LAM
    # Array factor (uniform weights)
    phase_arg = k * d * (np.sin(theta_rad) - np.sin(steer_rad))
    # Numerically stable form: |sin(Nψ/2)/sin(ψ/2)|
    psi = phase_arg
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(
            np.abs(np.sin(psi / 2)) < 1e-10,
            float(N_elements),
            np.abs(np.sin(N_elements * psi / 2) / np.sin(psi / 2)),
        )
    # Element directivity for w = λ/2
    d_elem = piston_directivity(theta, k * d / 2.0)  # ka = k*(d/2)
    beam = 20 * np.log10(np.abs(af * d_elem) / N_elements + 1e-10)

    ax.plot(theta, beam, lw=1.2)
    ax.axhline(-6, color="k", lw=0.7, ls=":")
    ax.axhline(-20, color="grey", lw=0.6, ls=":")
    ax.axvline(steer_deg, color="C1", lw=1.0, ls="--", label=f"θ_s = {steer_deg}°")

    # Mark theoretical grating lobes
    for m in [-2, -1, 1, 2]:
        sin_gl = np.sin(steer_rad) + m * LAM / d
        if -1 <= sin_gl <= 1:
            gl_angle = np.rad2deg(np.arcsin(sin_gl))
            ax.axvline(gl_angle, color="red", lw=0.8, ls=":", alpha=0.7)
            ax.text(gl_angle, 2, f"m={m}", color="red", fontsize=7, ha="center")

    ax.set_title(f"$d = {d_ratio}\\lambda$\n{'(grating lobe)' if d_ratio > 0.5 else '(GL-free)'}")
    ax.set_xlabel(r"Angle $\theta$ (°)")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

axes[0].set_ylabel("Beam pattern (dB, norm.)")
fig.suptitle(
    f"Array Factor × Element Directivity — {N_elements} elements, $f_0={F0/1e6:.0f}$ MHz,"
    f" $\\theta_s = {steer_deg}°$",
    y=1.02,
)
plt.tight_layout()
savefig("fig02_array_factor_grating_lobes")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 03: Apodization windows and frequency response (sidelobe comparison)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig03] Apodization windows and frequency response")

N_ap = 64
n = np.arange(N_ap)

windows = {
    "Uniform": np.ones(N_ap),
    "Hann": np.hanning(N_ap),
    "Hamming": np.hamming(N_ap),
    "Blackman": np.blackman(N_ap),
    "Tukey (r=0.5)": np.where(
        n < N_ap // 4,
        0.5 * (1 - np.cos(2 * np.pi * n / (N_ap // 2))),
        np.where(n > 3 * N_ap // 4, 0.5 * (1 - np.cos(2 * np.pi * (N_ap - 1 - n) / (N_ap // 2))), 1.0),
    ),
}
colors_w = ["C0", "C1", "C2", "C3", "C4"]

NFFT = 4096
freq_ax = np.linspace(-0.5, 0.5, NFFT)  # normalized frequency (cycles / sample)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for (name, w), col in zip(windows.items(), colors_w):
    ax1.plot(n, w / w.max(), label=name, color=col)
    W = np.fft.fftshift(np.fft.fft(w, n=NFFT))
    W_dB = 20 * np.log10(np.abs(W) / np.abs(W).max() + 1e-12)
    ax2.plot(freq_ax * N_ap, W_dB, label=name, color=col)

ax1.set_xlabel("Element index $n$")
ax1.set_ylabel("Normalized weight $w_n$")
ax1.set_title("Apodization Windows")
ax1.legend(fontsize=8)

ax2.set_xlabel("Spatial frequency (cycles / aperture)")
ax2.set_ylabel("Response (dB)")
ax2.set_title("Frequency Response (sidelobe envelope)")
ax2.set_xlim(-8, 8)
ax2.set_ylim(-80, 5)
ax2.axhline(-6, color="k", lw=0.6, ls=":", label="−6 dB")
ax2.axhline(-13.2, color="grey", lw=0.5, ls="--", label="Uniform PSL")
ax2.legend(fontsize=7, ncol=2)

plt.tight_layout()
savefig("fig03_apodization_windows")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04: Lateral resolution vs depth for several array apertures
# ─────────────────────────────────────────────────────────────────────────────

print("[fig04] Lateral resolution vs depth")

depths_mm = np.linspace(10, 80, 200)
apertures_mm = [10, 20, 38.4]   # 64 elem × 0.15mm / 0.3mm / 0.6mm pitch

fig, ax = plt.subplots(figsize=(7, 4.5))
colors_a = ["C0", "C1", "C2"]
for L_mm, col in zip(apertures_mm, colors_a):
    L = L_mm * 1e-3
    F_num = depths_mm * 1e-3 / L
    delta_x = 0.886 * LAM * F_num * 1e3  # convert to mm
    ax.plot(depths_mm, delta_x, color=col, label=f"$L = {L_mm}$ mm")

ax.set_xlabel("Depth (mm)")
ax.set_ylabel(r"Lateral resolution $\Delta x_{-6\,\mathrm{dB}}$ (mm)")
ax.set_title(
    r"Lateral Resolution vs Depth (Eq. 4.12, $f_0 = 5\,\mathrm{MHz}$, uniform apodization)"
)
ax.legend()
ax.set_xlim(10, 80)
ax.set_ylim(0)
ax.grid(True, ls=":", alpha=0.4)
plt.tight_layout()
savefig("fig04_lateral_resolution_vs_depth")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05: 2-D beam pattern (azimuth vs depth) for a focused linear array
# ─────────────────────────────────────────────────────────────────────────────

print("[fig05] 2-D beam pattern map")

N_el = 64
D_el = 0.3e-3       # element pitch [m] ≈ 0.97 λ
L = N_el * D_el     # aperture [m]
Z_F = 0.04          # focal depth [m]

k = 2 * np.pi / LAM
x_pts = np.linspace(-10e-3, 10e-3, 400)   # transverse [m]
z_pts = np.linspace(5e-3, 60e-3, 400)     # depth [m]

# Compute coherent pressure field using paraxial DAS
# p(x,z) ∝ Σ_n exp(ik|r_n − (x,z)|) · exp(−ik|r_n − focus|)
# approximated in the far field as exp(ik Δz) × sinc aperture factor
X, Z = np.meshgrid(x_pts, z_pts)

# Array factor at each field point (far-field approximation)
el_positions = (np.arange(N_el) - (N_el - 1) / 2) * D_el
# Hann apodization
w_hann = np.hanning(N_el)

# Transmit (focus at Z_F, θ_s=0)
# Receive from each point with same weights
p_field = np.zeros_like(X, dtype=complex)
for xi, w in zip(el_positions, w_hann):
    # path from element to field point
    dist_tx = np.sqrt((X - xi)**2 + Z**2)
    # transmit delay: distance to focus
    dist_focus = np.sqrt(xi**2 + Z_F**2)
    focus_delay = (dist_tx - dist_focus) / C0
    p_field += w * np.exp(1j * k * (dist_tx - dist_focus))

p_dB = 20 * np.log10(np.abs(p_field) / np.abs(p_field).max() + 1e-12)
p_dB = np.clip(p_dB, -60, 0)

fig, ax = plt.subplots(figsize=(6, 7))
pcm = ax.pcolormesh(
    x_pts * 1e3, z_pts * 1e3, p_dB,
    cmap="gray_r", vmin=-60, vmax=0, shading="auto"
)
ax.axhline(Z_F * 1e3, color="cyan", lw=0.8, ls="--", label=f"Focus z = {Z_F*1e3:.0f} mm")
ax.set_xlabel("Lateral position (mm)")
ax.set_ylabel("Depth (mm)")
ax.invert_yaxis()
ax.set_title(
    f"2-D Beam Pattern — {N_el}-element linear array\n"
    f"Hann apodization, $f_0 = {F0/1e6:.0f}$ MHz, $d = {D_el*1e3:.1f}$ mm"
)
ax.legend(loc="upper right", fontsize=8)
plt.colorbar(pcm, ax=ax, label="Pressure (dB)", fraction=0.03)
plt.tight_layout()
savefig("fig05_beam_pattern_2d")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 06: BLI stencil weights vs off-grid offset
# ─────────────────────────────────────────────────────────────────────────────

print("[fig06] BLI stencil weights")

# Stencil half-width: N_sub = ceil(1/(π ε)) for ε = 0.05 → N_sub = 7
EPS_BLI = 0.05
N_sub = int(np.ceil(1.0 / (np.pi * EPS_BLI)))  # = 7

# Off-grid offsets Δx/Δgrid ∈ [0, 0.5]
offsets = np.linspace(0, 0.5, 5)  # fractional grid offsets
grid_idx = np.arange(-N_sub, N_sub + 1)

fig, axes = plt.subplots(1, len(offsets), figsize=(14, 3.5), sharey=True)
for ax, frac_offset in zip(axes, offsets):
    # BLI weight: sinc(π(n − frac_offset))
    x = np.pi * (grid_idx - frac_offset)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / x)
    markerline, stemline, baseline = ax.stem(
        grid_idx, weights, linefmt="C0-", markerfmt="C0o", basefmt="k-"
    )
    plt.setp(markerline, markersize=4)
    ax.set_title(f"$\\Delta x / \\Delta_{{\\rm grid}} = {frac_offset:.1f}$", fontsize=9)
    ax.set_xlabel("Grid index $i$")
    ax.axhline(EPS_BLI, color="r", lw=0.7, ls=":", label=f"ε = {EPS_BLI}")
    ax.axhline(-EPS_BLI, color="r", lw=0.7, ls=":")
    ax.set_xlim(-N_sub - 0.5, N_sub + 0.5)

axes[0].set_ylabel("BLI weight $w_i$")
fig.suptitle(
    f"BLI Stencil Weights (Eq. 4.18) — $N_{{\\rm sub}} = {N_sub}$, $\\varepsilon_{{\\rm BLI}} = {EPS_BLI}$",
    y=1.02,
)
plt.tight_layout()
savefig("fig06_bli_stencil_weights")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(
    f"\nChapter 4 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_element_directivity.*       — Baffled piston D(θ) for ka = 1,2,4,8\n"
    "  fig02_array_factor_grating_lobes.*— 64-element AF × D, d/λ = 0.5, 1, 2\n"
    "  fig03_apodization_windows.*       — Window functions and frequency response\n"
    "  fig04_lateral_resolution_vs_depth.*— Δx vs depth for 3 apertures\n"
    "  fig05_beam_pattern_2d.*           — 2-D DAS beam pattern (Hann, 64 elem)\n"
    "  fig06_bli_stencil_weights.*       — BLI sinc stencil for 5 fractional offsets\n"
)
