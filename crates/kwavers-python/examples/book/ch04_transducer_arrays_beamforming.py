"""
Chapter 4: Transducer Arrays and Beamforming — Figure Generation Script
========================================================================

Generates all publication-quality figures for Chapter 4 of the kwavers book.
All physics computed by kwavers (Rust); this file contains only matplotlib
rendering.  Requires pykwavers to be installed.

Output directory: docs/book/figures/ch04/

References
----------
- O'Neil (1949) baffled piston directivity
- Szabo (2014) Diagnostic Ultrasound Imaging
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch04")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch04/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
    "figure.dpi": 150,
})

C0 = 1540.0
F0 = 5.0e6
LAM = C0 / F0

print("[fig01] Baffled piston element directivity")

theta = np.linspace(-90.0, 90.0, 1800)
theta_rad = np.deg2rad(theta)
ka_vals = [1.0, 2.0, 4.0, 8.0]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(ka_vals)))

fig, ax = plt.subplots(figsize=(7, 4.5))
for ka, col in zip(ka_vals, colors):
    d = kw.circular_piston_directivity(theta_rad, ka)
    d_dB = 20 * np.log10(np.abs(d) + 1e-12)
    ax.plot(theta, d_dB, color=col, label=f"$ka = {ka}$")

ax.axhline(-6, color="k", lw=0.8, ls=":", label="-6 dB")
ax.set_xlabel(r"Angle $\theta$ (deg)")
ax.set_ylabel(r"$|D(\theta)|$ (dB)")
ax.set_title("Baffled Circular Piston Directivity (Eq. 4.3)")
ax.set_xlim(-90, 90)
ax.set_ylim(-60, 2)
ax.legend(ncol=2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.tight_layout()
savefig("fig01_element_directivity")
plt.close()

print("[fig02] Array factor and grating lobes")

N_elements = 64
pitches_lambda = [0.5, 1.0, 2.0]
steer_deg = 20.0
steer_rad = np.deg2rad(steer_deg)
theta = np.linspace(-90.0, 90.0, 3600)
theta_rad = np.deg2rad(theta)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, d_ratio in zip(axes, pitches_lambda):
    d = d_ratio * LAM
    k = 2 * np.pi / LAM
    af = kw.linear_array_factor(N_elements, d, LAM, theta_rad, steer_rad)
    d_elem = kw.circular_piston_directivity(theta_rad, k * d / 2.0)
    beam = 20 * np.log10(np.abs(af * d_elem) / N_elements + 1e-10)

    ax.plot(theta, beam, lw=1.2)
    ax.axhline(-6, color="k", lw=0.7, ls=":")
    ax.axhline(-20, color="grey", lw=0.6, ls=":")
    ax.axvline(steer_deg, color="C1", lw=1.0, ls="--", label=f"theta_s = {steer_deg} deg")

    for m in [-2, -1, 1, 2]:
        sin_gl = np.sin(steer_rad) + m * LAM / d
        if -1 <= sin_gl <= 1:
            gl_angle = np.rad2deg(np.arcsin(sin_gl))
            ax.axvline(gl_angle, color="red", lw=0.8, ls=":", alpha=0.7)
            ax.text(gl_angle, 2, f"m={m}", color="red", fontsize=7, ha="center")

    ax.set_title(f"$d = {d_ratio}\\lambda$\n{'(grating lobe)' if d_ratio > 0.5 else '(GL-free)'}")
    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

axes[0].set_ylabel("Beam pattern (dB, norm.)")
fig.suptitle(
    f"Array Factor x Element Directivity - {N_elements} elements, $f_0={F0/1e6:.0f}$ MHz,"
    f" steer = {steer_deg} deg",
    y=1.02,
)
plt.tight_layout()
savefig("fig02_array_factor_grating_lobes")
plt.close()

print("[fig03] Apodization windows and frequency response")

N_ap = 64
n = np.arange(N_ap)
window_names = ["Uniform", "Hann", "Hamming", "Blackman", "Tukey_05"]
colors_w = ["C0", "C1", "C2", "C3", "C4"]
windows_data = {}
for wname in window_names:
    windows_data[wname] = kw.apodization_weights(N_ap, wname)

NFFT = 4096
freq_ax = np.linspace(-0.5, 0.5, NFFT)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for (wname, w), col in zip(windows_data.items(), colors_w):
    label = wname.replace("_05", " (r=0.5)")
    ax1.plot(n, w / w.max(), label=label, color=col)
    W = np.fft.fftshift(np.fft.fft(w, n=NFFT))
    W_dB = 20 * np.log10(np.abs(W) / np.abs(W).max() + 1e-12)
    ax2.plot(freq_ax * N_ap, W_dB, label=label, color=col)

ax1.set_xlabel("Element index $n$")
ax1.set_ylabel("Normalized weight $w_n$")
ax1.set_title("Apodization Windows")
ax1.legend(fontsize=8)
ax2.set_xlabel("Spatial frequency (cycles / aperture)")
ax2.set_ylabel("Response (dB)")
ax2.set_title("Frequency Response (sidelobe envelope)")
ax2.set_xlim(-8, 8)
ax2.set_ylim(-80, 5)
ax2.axhline(-6, color="k", lw=0.6, ls=":", label="-6 dB")
ax2.legend(fontsize=7, ncol=2)
plt.tight_layout()
savefig("fig03_apodization_windows")
plt.close()

print("[fig04] Lateral resolution vs depth")

depths_mm = np.linspace(10, 80, 200)
apertures_mm = [10, 20, 38.4]

fig, ax = plt.subplots(figsize=(7, 4.5))
for L_mm, col in zip(apertures_mm, ["C0", "C1", "C2"]):
    delta_x_mm = kw.lateral_resolution_m(depths_mm * 1e-3, L_mm * 1e-3, LAM) * 1e3
    ax.plot(depths_mm, delta_x_mm, color=col, label=f"$L = {L_mm}$ mm")

ax.set_xlabel("Depth (mm)")
ax.set_ylabel(r"Lateral resolution $\Delta x_{-6\,\mathrm{dB}}$ (mm)")
ax.set_title(r"Lateral Resolution vs Depth ($f_0 = 5\,\mathrm{MHz}$, uniform apodization)")
ax.legend()
ax.set_xlim(10, 80)
ax.set_ylim(0)
ax.grid(True, ls=":", alpha=0.4)
plt.tight_layout()
savefig("fig04_lateral_resolution_vs_depth")
plt.close()

print("[fig05] 2-D beam pattern map")

N_el = 64
D_el = 0.3e-3
L = N_el * D_el
Z_F = 0.04
x_pts = np.linspace(-10e-3, 10e-3, 400)
z_pts = np.linspace(5e-3, 60e-3, 400)
X, Z = np.meshgrid(x_pts, z_pts)

el_positions = (np.arange(N_el) - (N_el - 1) / 2.0) * D_el
w_hann = kw.apodization_weights(N_el, "Hann")
delays = kw.delay_law_focus_2d(el_positions, 0.0, Z_F, C0)

p_field = kw.beam_pattern_2d(X.ravel(), Z.ravel(), el_positions, w_hann, delays, C0, F0)
p_field = p_field.reshape(X.shape)
p_dB = 20 * np.log10(np.abs(p_field) / np.abs(p_field).max() + 1e-12)
p_dB = np.clip(p_dB, -60, 0)

fig, ax = plt.subplots(figsize=(6, 7))
pcm = ax.pcolormesh(x_pts * 1e3, z_pts * 1e3, p_dB,
                    cmap="gray_r", vmin=-60, vmax=0, shading="auto")
ax.axhline(Z_F * 1e3, color="cyan", lw=0.8, ls="--", label=f"Focus z = {Z_F*1e3:.0f} mm")
ax.set_xlabel("Lateral position (mm)")
ax.set_ylabel("Depth (mm)")
ax.invert_yaxis()
ax.set_title(
    f"2-D Beam Pattern - {N_el}-element linear array\n"
    f"Hann apodization, $f_0 = {F0/1e6:.0f}$ MHz, $d = {D_el*1e3:.1f}$ mm"
)
ax.legend(loc="upper right", fontsize=8)
plt.colorbar(pcm, ax=ax, label="Pressure (dB)", fraction=0.03)
plt.tight_layout()
savefig("fig05_beam_pattern_2d")
plt.close()

print("[fig06] BLI stencil weights")

EPS_BLI = 0.05
N_sub = int(np.ceil(1.0 / (np.pi * EPS_BLI)))
offsets = np.linspace(0, 0.5, 5)
grid_idx = np.arange(-N_sub, N_sub + 1)

fig, axes = plt.subplots(1, len(offsets), figsize=(14, 3.5), sharey=True)
for ax, frac_offset in zip(axes, offsets):
    weights = kw.bli_stencil_weights(grid_idx.astype(float), frac_offset, EPS_BLI)
    markerline, stemline, baseline = ax.stem(
        grid_idx, weights, linefmt="C0-", markerfmt="C0o", basefmt="k-"
    )
    plt.setp(markerline, markersize=4)
    ax.set_title(f"offset = {frac_offset:.1f}", fontsize=9)
    ax.set_xlabel("Grid index $i$")
    ax.axhline(EPS_BLI, color="r", lw=0.7, ls=":")
    ax.axhline(-EPS_BLI, color="r", lw=0.7, ls=":")
    ax.set_xlim(-N_sub - 0.5, N_sub + 0.5)

axes[0].set_ylabel("BLI weight $w_i$")
fig.suptitle(f"BLI Stencil Weights (Eq. 4.18) - N_sub = {N_sub}, eps = {EPS_BLI}", y=1.02)
plt.tight_layout()
savefig("fig06_bli_stencil_weights")
plt.close()

print(f"\nChapter 4 figures written to: {os.path.relpath(OUT_DIR)}\n")
