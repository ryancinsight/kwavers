"""
Chapter 6: Therapeutic Ultrasound — Figure Generation Script
=============================================================

Figures produced:
  fig01: Pennes bioheat temperature rise vs depth — real ThermalDiffusionSolver (Eq. 6.3)
  fig02: CEM43 accumulation vs temperature and duration — pykwavers.cem43_at_temperatures
  fig03: HIFU focal intensity gain vs f-number (Theorem 4.9, Eq. 6.2)
  fig04: Acoustic absorption in tissue — power deposition depth profile (Eq. 6.1)
  fig05: Thermal ablation zone — real 2-D ThermalDiffusionSolver with Pennes bioheat

Physics:
  All thermal diffusion and Pennes bioheat computations are delegated to
  kwavers::solver::forward::thermal_diffusion::ThermalDiffusionSolver via
  pykwavers.ThermalSimulation.  Python is used only for plotting.

Output directory: docs/book/figures/ch06/
Requires: numpy, matplotlib, pykwavers
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

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

# Physical constants — liver tissue (ICRU Report 44)
C0 = 1540.0       # sound speed [m/s]
RHO0 = 1060.0     # density [kg/m³]
CP = 3600.0        # specific heat [J/(kg·K)]
K_LIVER = 0.51    # thermal conductivity [W/(m·K)]
ALPHA_ABS = 7.0   # acoustic absorption [Np/m] at 1 MHz
F0 = 1.0e6        # driving frequency [Hz]
WB = 5e-3         # blood perfusion rate [1/s] (liver)
RHO_B = 1050.0    # blood density [kg/m³]
CPB = 3840.0      # blood specific heat [J/(kg·K)]
T_ART = 37.0      # arterial temperature [°C]

# Thermal diffusion length scale (stability reference)
D_LIVER = K_LIVER / (RHO0 * CP)  # ≈ 1.34e-7 m²/s

# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: Pennes bioheat temperature rise vs time — ThermalDiffusionSolver
# Q = 2·α·I at the focal point; includes thermal diffusion + blood perfusion.
# ─────────────────────────────────────────────────────────────────────────────

print("[fig01] Pennes bioheat temperature rise (ThermalDiffusionSolver)")

if not _HAS_PYKWAVERS:
    raise ImportError("pykwavers is required for fig01 (ThermalDiffusionSolver)")

# 1-D slab: 200 cells × 0.5 mm = 100 mm domain
NX_T1 = 200
DX_T1 = 5e-4       # 0.5 mm
DT_T1 = 0.05       # s  (dt_max_1D = DX²/(2D) ≈ 0.93 s)
N_T1 = 100          # → 5 s total

IX_FOC = NX_T1 // 2  # focal point index

sim1 = kw.ThermalSimulation(
    NX_T1, 1, 1, DX_T1, DX_T1, DX_T1,
    thermal_conductivity=K_LIVER,
    density=RHO0,
    specific_heat=CP,
    enable_bioheat=True,
    perfusion_rate=WB,
    blood_density=RHO_B,
    blood_specific_heat=CPB,
    arterial_temperature=T_ART,
    initial_temperature=T_ART,
    track_thermal_dose=False,
)

# Focal heat source: Gaussian in depth, σ = 4 cells (2 mm), Q_peak = 2·α·I
ix_all = np.arange(NX_T1)
heat_env = np.exp(-0.5 * ((ix_all - IX_FOC) / 4.0) ** 2)

sensor_mask1 = np.zeros((NX_T1, 1, 1), dtype=bool)
sensor_mask1[IX_FOC, 0, 0] = True

intensities = [1e7, 5e7, 1e8, 5e8]  # W/m²

fig, ax = plt.subplots(figsize=(7, 4.5))
for I in intensities:
    Q_field = np.zeros((NX_T1, 1, 1))
    Q_field[:, 0, 0] = 2.0 * ALPHA_ABS * I * heat_env
    res1 = sim1.run(N_T1, DT_T1, heat_source=Q_field, sensor_mask=sensor_mask1)
    t_s = np.asarray(res1.time)
    T_t = np.asarray(res1.temperature_at_sensors)[0, :]  # focal sensor, shape (N_T1,)
    label = f"$I = {I/1e4:.0f}$ W/cm²"
    ax.plot(t_s, T_t, label=label)

ax.axhline(60.0, color="r", lw=1.0, ls="--", label="Protein denaturation (60°C)")
ax.axhline(43.0, color="orange", lw=0.8, ls=":", label="CEM43 reference (43°C)")
ax.set_xlabel("Exposure duration (s)")
ax.set_ylabel("Focal temperature (°C)")
ax.set_title(
    f"Focal Temperature Rise — Liver, $f_0 = {F0/1e6:.0f}$ MHz\n"
    "(Eq. 6.4, Pennes bioheat: diffusion + perfusion; kwavers ThermalDiffusionSolver)"
)
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(0, N_T1 * DT_T1)
ax.set_ylim(35)
ax.grid(True, ls=":", alpha=0.4)
plt.tight_layout()
savefig("fig01_temperature_rise")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: CEM43 accumulation — pykwavers.cem43_at_temperatures
# ─────────────────────────────────────────────────────────────────────────────

print("[fig02] CEM43 accumulation (pykwavers.cem43_at_temperatures)")

T_arr = np.linspace(37, 70, 300)  # °C
durations_s = [1, 5, 10, 60]     # seconds

fig, ax = plt.subplots(figsize=(7, 4.5))
for dur in durations_s:
    # CEM43 = R^{43−T} · Δt/60 (Sapareto & Dewey 1984, Eq. 6.7)
    CEM43 = np.asarray(kw.cem43_at_temperatures(T_arr, float(dur)))
    ax.semilogy(T_arr, np.maximum(CEM43, 1e-6), label=f"$t = {dur}$ s")

ax.axhline(240, color="r", lw=1.2, ls="--", label="Ablation threshold (240 min)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("CEM43 (min)")
ax.set_title("Thermal Dose Accumulation (Eq. 6.7, Sapareto & Dewey 1984)\n"
             "Computed by pykwavers.cem43_at_temperatures")
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
# Analytical: G = k·a²/(2·R_f)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig03] HIFU focal gain vs f-number")

LAM = C0 / F0  # wavelength
k = 2 * np.pi / LAM

a_vals_mm = [20, 30, 40]    # aperture radii [mm]
Rf_arr = np.linspace(30e-3, 150e-3, 200)  # focal lengths [m]

fig, ax = plt.subplots(figsize=(7, 4.5))
for a_mm in a_vals_mm:
    a = a_mm * 1e-3
    G = k * a ** 2 / (2 * Rf_arr)
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
# Beer-Lambert: I(z) = I₀·exp(−2αz), Q(z) = 2α·I(z)
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
# Figure 05: 2-D thermal ablation zone — ThermalDiffusionSolver with Pennes bioheat
# Grid: 160 × 1 × 160 cells, dx = dz = 0.5 mm → 80 mm × 80 mm plane.
# Heat source: Gaussian focal pattern Q = 2·α·I·exp(−r²), representing HIFU
# intensity at 500 MW/m² (5 kW/cm²).  ThermalDiffusionSolver propagates heat
# for 1 s and accumulates the CEM43 dose field.
# ─────────────────────────────────────────────────────────────────────────────

print("[fig05] 2-D thermal ablation zone (ThermalDiffusionSolver, 1 s sonication)")

NX_T5 = 160   # depth (axial) cells  — 80 mm
NZ_T5 = 160   # lateral cells        — 80 mm
DX_T5 = 5e-4  # 0.5 mm
# Stability: dt_max_2D = DX²/(4·D) = (5e-4)²/(4·1.34e-7) ≈ 0.47 s → dt=0.05 s is safe
DT_T5 = 0.05  # s
N_T5 = 20     # → 1 s total sonication

IX_FOC5 = 120  # depth focus: 60 mm
IZ_FOC5 = NZ_T5 // 2  # lateral center

# Gaussian focal heat source: Q_peak = 2·α·I; σ_axial = 5 cells (2.5 mm), σ_lat = 2 (1 mm)
IX5, IZ5 = np.meshgrid(np.arange(NX_T5), np.arange(NZ_T5), indexing="ij")
I_FOCAL = 5e8   # W/m², focal intensity
Q_env = (
    2.0 * ALPHA_ABS * I_FOCAL
    * np.exp(
        -0.5 * ((IX5 - IX_FOC5) / 5.0) ** 2
        - 0.5 * ((IZ5 - IZ_FOC5) / 2.0) ** 2
    )
)
Q_field5 = Q_env[:, np.newaxis, :]  # shape (NX_T5, 1, NZ_T5)

sim5 = kw.ThermalSimulation(
    NX_T5, 1, NZ_T5, DX_T5, DX_T5, DX_T5,
    thermal_conductivity=K_LIVER,
    density=RHO0,
    specific_heat=CP,
    enable_bioheat=True,
    perfusion_rate=WB,
    blood_density=RHO_B,
    blood_specific_heat=CPB,
    arterial_temperature=T_ART,
    initial_temperature=T_ART,
    track_thermal_dose=True,
)
res5 = sim5.run(N_T5, DT_T5, heat_source=Q_field5)

# Extract 2-D slices at ny=0
T_map = np.asarray(res5.temperature)[:, 0, :]     # (NX_T5, NZ_T5) [°C]
dose_map = np.asarray(res5.thermal_dose)[:, 0, :] # (NX_T5, NZ_T5) [min]

depth_mm = np.arange(NX_T5) * DX_T5 * 1e3          # 0…80 mm
lateral_mm = (np.arange(NZ_T5) - IZ_FOC5) * DX_T5 * 1e3  # ±40 mm

# Plot
cmap_therapy = LinearSegmentedColormap.from_list(
    "therapy",
    ["#4466cc", "#88aaff", "#ffffff", "#ffaa44", "#cc2200"],
)

fig, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(12, 7))

pcm_t = ax_t.pcolormesh(
    lateral_mm, depth_mm, T_map, cmap=cmap_therapy, vmin=37, vmax=100, shading="auto"
)
cs_t = ax_t.contour(
    lateral_mm, depth_mm, T_map,
    levels=[43, 60, 80], colors=["orange", "red", "darkred"],
    linewidths=[1.0, 1.5, 1.0], linestyles=["--", "-", ":"]
)
ax_t.clabel(cs_t, fmt="%.0f°C", fontsize=8)
ax_t.set_xlabel("Lateral (mm)")
ax_t.set_ylabel("Depth (mm)")
ax_t.invert_yaxis()
ax_t.set_title("Temperature — HIFU, 1 s sonication\n(ThermalDiffusionSolver + Pennes bioheat)")
plt.colorbar(pcm_t, ax=ax_t, label="Temperature (°C)", fraction=0.03)

pcm_d = ax_d.pcolormesh(
    lateral_mm, depth_mm, np.log10(np.maximum(dose_map, 1e-4)),
    cmap="inferno", shading="auto"
)
cs_d = ax_d.contour(
    lateral_mm, depth_mm, dose_map,
    levels=[60, 240], colors=["yellow", "white"],
    linewidths=[1.0, 1.5]
)
ax_d.clabel(cs_d, fmt="%.0f min", fontsize=8)
ax_d.set_xlabel("Lateral (mm)")
ax_d.set_ylabel("Depth (mm)")
ax_d.invert_yaxis()
ax_d.set_title("CEM43 Thermal Dose [log₁₀ min]\nDamage (60 min) and ablation (240 min) contours")
plt.colorbar(pcm_d, ax=ax_d, label="log₁₀ CEM43 (min)", fraction=0.03)

plt.tight_layout()
savefig("fig05_ablation_zone_2d")
plt.close()

print(
    f"\nChapter 6 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_temperature_rise.*        — Pennes bioheat T(t) at focus; 4 intensities\n"
    "  fig02_cem43_accumulation.*      — CEM43(T) via pykwavers.cem43_at_temperatures\n"
    "  fig03_hifu_focal_gain.*         — Focal gain G (dB) vs f-number [analytical]\n"
    "  fig04_power_deposition_profile.* — I(z) and Q(z) vs depth [Beer-Lambert]\n"
    "  fig05_ablation_zone_2d.*        — 2-D T and CEM43 from ThermalDiffusionSolver\n"
)
