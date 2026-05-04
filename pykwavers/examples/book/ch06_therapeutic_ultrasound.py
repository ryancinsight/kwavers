"""
Chapter 6: Therapeutic Ultrasound — Figure Generation Script
=============================================================

Figures produced:
  fig01: Bioheat temperature rise vs depth — Pennes equation (Eq. 6.3)
  fig02: CEM43 accumulation vs temperature and duration (Eq. 6.7)
  fig03: HIFU focal intensity gain vs f-number (Theorem 4.9, Eq. 6.2)
  fig04: Acoustic absorption in tissue — power deposition depth profile (Eq. 6.1)
  fig05: Thermal ablation zone — temperature iso-surface (2-D Gaussian approximation)

Output directory: docs/book/figures/ch06/
Requires: numpy, matplotlib, scipy
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch06")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch06/{name}.{{pdf,png}}")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
    }
)

# Physical constants
C0 = 1540.0      # m/s
RHO0 = 1060.0    # kg/m³ (liver)
CP = 3600.0      # J/kg/K (liver)
KAPPA = 0.51     # W/m/K (liver thermal conductivity)
ALPHA_ABS = 7.0  # Np/m (liver, 1 MHz)
F0 = 1.0e6       # Hz

# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: Temperature rise vs exposure duration for several focal intensities
# ─────────────────────────────────────────────────────────────────────────────

print("[fig01] Temperature rise vs exposure duration")

t_arr = np.linspace(0, 5.0, 300)  # seconds
intensities = [1e7, 5e7, 1e8, 5e8]  # W/m²

fig, ax = plt.subplots(figsize=(7, 4.5))
for I in intensities:
    Q = 2 * ALPHA_ABS * I  # W/m³
    dT_dt = Q / (RHO0 * CP)  # °C/s
    T = 37.0 + dT_dt * t_arr  # initial temperature 37°C
    label = f"$I = {I/1e4:.0f}$ W/cm²"
    ax.plot(t_arr, T, label=label)

ax.axhline(60.0, color="r", lw=1.0, ls="--", label="Protein denaturation (60°C)")
ax.axhline(43.0, color="orange", lw=0.8, ls=":", label="CEM43 reference (43°C)")
ax.set_xlabel("Exposure duration (s)")
ax.set_ylabel("Temperature (°C)")
ax.set_title(
    f"Focal Temperature Rise — Liver Tissue, $f_0 = {F0/1e6:.0f}$ MHz\n"
    "(Eq. 6.4, no perfusion/conduction)"
)
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(0, 5)
ax.set_ylim(35, 600)
ax.grid(True, ls=":", alpha=0.4)
plt.tight_layout()
savefig("fig01_temperature_rise")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: CEM43 accumulation — dose vs temperature for fixed durations
# ─────────────────────────────────────────────────────────────────────────────

print("[fig02] CEM43 accumulation")

T_arr = np.linspace(37, 70, 300)  # °C
durations_s = [1, 5, 10, 60]  # seconds

fig, ax = plt.subplots(figsize=(7, 4.5))
for dur in durations_s:
    R = np.where(T_arr >= 43, 0.5, 0.25)
    CEM43 = dur / 60.0 * np.power(R, 43 - T_arr)  # minutes
    ax.semilogy(T_arr, np.maximum(CEM43, 1e-6), label=f"$t = {dur}$ s")

ax.axhline(240, color="r", lw=1.2, ls="--", label="Ablation threshold (240 min)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("CEM43 (min)")
ax.set_title("Thermal Dose Accumulation (Eq. 6.7, Sapareto & Dewey 1984)")
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(37, 70)
ax.set_ylim(1e-4, 1e6)
ax.axvline(43, color="grey", lw=0.7, ls=":")
ax.text(43.3, 5e5, "43°C", color="grey", fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.3)
plt.tight_layout()
savefig("fig02_cem43_accumulation")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 03: HIFU focal intensity gain vs f-number (Theorem 4.9 / Eq. 6.2)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig03] HIFU focal gain vs f-number")

LAM = C0 / F0  # wavelength
k = 2 * np.pi / LAM

a_vals_mm = [20, 30, 40]  # aperture radii [mm]
Rf_arr = np.linspace(30e-3, 150e-3, 200)  # focal lengths [m]

fig, ax = plt.subplots(figsize=(7, 4.5))
for a_mm in a_vals_mm:
    a = a_mm * 1e-3
    G = k * a**2 / (2 * Rf_arr)
    G_dB = 20 * np.log10(G)
    F_num = Rf_arr / (2 * a)
    ax.plot(F_num, G_dB, label=f"$a = {a_mm}$ mm")

ax.set_xlabel("f-number $F\\# = R_f/(2a)$")
ax.set_ylabel("Focal gain $G$ (dB)")
ax.set_title(f"HIFU Focal Gain (Eq. 6.2) — $f_0 = {F0/1e6:.0f}$ MHz")
ax.legend()
ax.grid(True, ls=":", alpha=0.4)
ax.set_xlim(F_num[0], F_num[-1])
plt.tight_layout()
savefig("fig03_hifu_focal_gain")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04: Acoustic power deposition profile in tissue (Eq. 6.1)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig04] Power deposition depth profile")

z_arr = np.linspace(0, 80e-3, 400)  # m
I0 = 1.0  # normalized surface intensity

alpha_tissues = {
    "Liver (7 Np/m)": 7.0,
    "Breast fat (2 Np/m)": 2.0,
    "Skull bone (50 Np/m)": 50.0,
}
colors_t = ["C0", "C1", "C2"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for (label, alpha_t), col in zip(alpha_tissues.items(), colors_t):
    I_z = I0 * np.exp(-2 * alpha_t * z_arr)
    Q_z = 2 * alpha_t * I_z
    ax1.plot(z_arr * 1e3, I_z, color=col, label=label)
    ax2.plot(z_arr * 1e3, Q_z / (2 * alpha_t * I0), color=col, label=label)

ax1.set_xlabel("Depth z (mm)")
ax1.set_ylabel("Normalized intensity $I/I_0$")
ax1.set_title("Acoustic Intensity vs Depth")
ax1.legend(fontsize=8)
ax1.set_xlim(0, 80)
ax1.set_ylim(0)

ax2.set_xlabel("Depth z (mm)")
ax2.set_ylabel("Normalized heat source $Q/(2\\alpha I_0)$")
ax2.set_title("Volumetric Power Deposition (Eq. 6.1)")
ax2.legend(fontsize=8)
ax2.set_xlim(0, 80)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
savefig("fig04_power_deposition_profile")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05: 2-D thermal ablation zone (Gaussian temperature profile)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig05] 2-D thermal ablation zone")

NX, NZ = 200, 300
x = np.linspace(-15e-3, 15e-3, NX) * 1e3  # mm
z = np.linspace(0, 80e-3, NZ) * 1e3       # mm
X, Z = np.meshgrid(x, z)

# Gaussian focal heating approximation
z0_focus = 50.0   # mm, focal depth
sigma_lat = 1.0   # mm, lateral −6dB/2.35
sigma_ax = 2.5    # mm, axial

# Intensity profile
I_norm = np.exp(-0.5 * ((X / sigma_lat)**2 + ((Z - z0_focus) / sigma_ax)**2))
# Temperature after 1 s sonication (no diffusion, approx)
Q = 2 * ALPHA_ABS * 5e8 * I_norm  # W/m³
dT = Q * 1.0 / (RHO0 * CP)       # °C
T = 37.0 + dT

# Smooth with thermal diffusion approximation
T_smooth = gaussian_filter(T, sigma=[3, 1])

# Custom colormap: blue → white → orange → red
cmap_therapy = LinearSegmentedColormap.from_list(
    "therapy",
    ["#4466cc", "#88aaff", "#ffffff", "#ffaa44", "#cc2200"],
)

fig, ax = plt.subplots(figsize=(6, 8))
pcm = ax.pcolormesh(x, z, T_smooth, cmap=cmap_therapy, vmin=37, vmax=100, shading="auto")
# Contour for ablation threshold 60°C and CEM43 reference 43°C
cs_ablation = ax.contour(x, z, T_smooth, levels=[43, 60, 80], colors=["orange", "red", "darkred"],
                          linewidths=[1.0, 1.5, 1.0], linestyles=["--", "-", ":"])
ax.clabel(cs_ablation, fmt="%.0f°C", fontsize=8)
ax.set_xlabel("Lateral position (mm)")
ax.set_ylabel("Depth (mm)")
ax.invert_yaxis()
ax.set_title(
    "Thermal Ablation Zone — HIFU, $f_0 = 1$ MHz\n"
    "Gaussian focal heating, 1 s sonication (qualitative)"
)
plt.colorbar(pcm, ax=ax, label="Temperature (°C)", fraction=0.03)
plt.tight_layout()
savefig("fig05_ablation_zone_2d")
plt.close()

print(
    f"\nChapter 6 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_temperature_rise.*     — T vs exposure at several focal intensities\n"
    "  fig02_cem43_accumulation.*   — CEM43 vs temperature for fixed durations\n"
    "  fig03_hifu_focal_gain.*      — Focal gain G (dB) vs f-number\n"
    "  fig04_power_deposition.*     — Intensity and heat source vs depth\n"
    "  fig05_ablation_zone_2d.*     — 2-D Gaussian temperature map with 43/60°C contours\n"
)
