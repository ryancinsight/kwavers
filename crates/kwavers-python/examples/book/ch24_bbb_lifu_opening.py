"""
Chapter 24: LIFU-Mediated Blood–Brain Barrier Opening
======================================================

Publication-quality figures for docs/book/bbb_lifu_opening.md.

Low-Intensity Focused Ultrasound (LIFU) combined with systemically injected
microbubbles (MBs) transiently opens the Blood–Brain Barrier (BBB) via
cavitation-mediated bioeffects.  This chapter covers the biophysical
mechanisms, parameter space for safe/effective opening, permeability
enhancement models, thermal safety, and feedback monitoring.

Figures produced
----------------
fig01  MB dynamics: R(t) from Keller–Miksis ODE under sub-threshold LIFU
fig02  LIFU safety parameter space: MI vs frequency (stable vs inertial)
fig03  BBB permeability enhancement vs acoustic dose (Evans blue model)
fig04  Thermal safety: focal temperature rise and CEM43 for LIFU parameters
fig05  CEUS contrast enhancement: peak signal vs MB concentration
fig06  BBB opening window: permeability vs time post-sonication

All bubble dynamics use pykwavers.solve_keller_miksis (Rust RK4).
Thermal safety uses pykwavers.ThermalSimulation (Pennes bioheat, Rust solver)
and pykwavers.compute_cem43 (Rust CEM43 accumulator).

Output directory: docs/book/figures/ch24/
Requires: numpy, matplotlib, pykwavers

References
----------
Hynynen et al. (2001) Radiology 220(3):640–646
McDannold et al. (2008) Ultrasound Med. Biol. 34(6):930–937
Deffieux & Konofagou (2010) Ultrasound Med. Biol. 36(7):1117–1126
Tung et al. (2010) Proc. Natl. Acad. Sci. 107(8):3699–3704
O'Reilly & Hynynen (2012) Radiology 263(1):96–106
Duck (1990) Physical Properties of Tissue. Academic Press.
Sapareto & Dewey (1984) Int. J. Radiat. Oncol. Biol. Phys. 10(6):787–800
Keller & Miksis (1980) J. Acoust. Soc. Am. 68(2):628–633
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pykwavers as kw
    _HAS_KW = True
except ImportError:
    _HAS_KW = False
    raise RuntimeError(
        "pykwavers not found — build with `maturin develop --release` "
        "from the pykwavers directory."
    )

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch24")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch24/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

# ── Physical parameters ──────────────────────────────────────────────────────
RHO_L = 998.0      # kg/m³  liquid density (blood-like)
MU_L  = 3.5e-3     # Pa·s   blood viscosity
SIGMA = 0.0056     # N/m    MB shell surface tension (coated bubble)
P0    = 101_325.0  # Pa     ambient pressure
PV    = 2338.0     # Pa     water vapour pressure at 20 °C
C_L   = 1540.0     # m/s    blood sound speed
KAPPA = 1.07       # polytropic index (gas + shell)
R0    = 1.5e-6     # m      MB equilibrium radius (1.5 µm — SonoVue-like)
# Shell parameters (Doinikov–Dayton 2007 neo-Hookean)
XI_S  = 1.5e-9     # Pa·s·m  shell viscosity parameter

# ── Figure 01: Keller–Miksis microbubble dynamics ────────────────────────────
print("[fig01] MB dynamics — Keller–Miksis ODE (pykwavers Rust RK4)")

f0 = 1.0e6   # Hz
# 10 acoustic cycles; 3 000 pts/cycle gives smooth waveform resolution
N_STEPS_KM = 30_000
T_END_KM = 10.0 / f0  # 10 µs

pressures_pa = [50e3, 150e3, 350e3]  # Pa  — sub-threshold LIFU range
colors = ["#2166ac", "#4dac26", "#d6604d"]
labels = [f"|p_a| = {p/1e3:.0f} kPa" for p in pressures_pa]

fig, axes = plt.subplots(len(pressures_pa), 1, figsize=(8, 7), sharex=True)
for ax, pa, col, lbl in zip(axes, pressures_pa, colors, labels):
    # Rust RK4 Keller–Miksis solver with coated-bubble shell viscosity (xi_s)
    time_s, R_m, _ = kw.solve_keller_miksis(
        R0, 0.0,         # R₀, Ṙ₀
        P0, pa,          # ambient pressure, acoustic driving amplitude
        f0,              # driving frequency [Hz]
        T_END_KM,        # integration end time [s]
        N_STEPS_KM,      # RK4 step count
        RHO_L, SIGMA, KAPPA, MU_L, PV, C_L,
        XI_S,            # shell viscosity [Pa·s·m]
    )
    t_us = np.asarray(time_s) * 1e6
    r_norm = np.asarray(R_m) / R0
    ax.plot(t_us, r_norm, color=col, lw=1.4, label=lbl)
    ax.axhline(1.0, color="#aaa", lw=0.6, ls="--")
    ax.set_ylabel("$R/R_0$")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)

axes[-1].set_xlabel("Time (µs)")
axes[0].set_title(
    f"Keller–Miksis MB dynamics — $R_0$ = {R0*1e6:.1f} µm, $f_0$ = {f0/1e6:.0f} MHz\n"
    "SonoVue-like shell; LIFU sub-threshold driving (pykwavers Rust RK4)"
)
fig.tight_layout()
savefig("fig01_keller_miksis_dynamics")
plt.close(fig)


# ── Figure 02: LIFU safety parameter space (MI vs frequency) ─────────────────
print("[fig02] LIFU safety parameter space: MI vs frequency")

# Blake threshold (inertial cavitation onset) for a free bubble:
#   p_Blake = P₀ sqrt(8σ/(3(P₀ + 2σ/R₀))) · (R₀/R_c)^{3κ/2}
# Simplified: MI_IC ≈ 0.45 / sqrt(f_MHz)  (FDA/IEC empirical limit for diagnostic)
# BBB opening with MBs: MI_BBB ≈ 0.3–0.5 at f = 0.2–2 MHz (O'Reilly 2012)
# Tissue damage: MI_dam ≈ 1.0–1.9  (conservative; depends on exposure time)

f_mhz = np.linspace(0.1, 3.0, 300)
mi_fda = 1.9 * np.ones_like(f_mhz)        # FDA derated MI limit
mi_ic_free = 0.45 / np.sqrt(f_mhz)        # inertial cavitation onset (free bubble)
mi_ic_mb = 0.18 / np.sqrt(f_mhz)          # inertial onset with MBs (~4× lower pressure)
mi_bbb_low = 0.20 * np.ones_like(f_mhz)   # lower bound for BBB opening
mi_bbb_high = 0.55 * np.ones_like(f_mhz)  # upper bound for BBB opening

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(f_mhz, mi_bbb_low, mi_bbb_high, alpha=0.18, color="#4dac26",
                label="BBB opening window (stable cavitation)")
ax.plot(f_mhz, mi_ic_mb, color="#d6604d", lw=1.8, ls="--",
        label="IC onset with MBs (× 4 sensitisation)")
ax.plot(f_mhz, mi_ic_free, color="#a50026", lw=1.4, ls=":",
        label="IC onset (free bubble, Blake threshold)")
ax.axhline(mi_fda[0], color="#555", lw=1.2, ls="-.",
           label=f"FDA MI limit ({mi_fda[0]:.1f})")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Mechanical Index (MI)")
ax.set_xlim(0.1, 3.0)
ax.set_ylim(0, 2.2)
ax.set_title("LIFU parameter space for BBB opening\n"
             "Green band: stable cavitation → reversible BBB permeabilisation")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
savefig("fig02_lifu_parameter_space")
plt.close(fig)


# ── Figure 03: BBB permeability vs acoustic dose ──────────────────────────────
print("[fig03] BBB permeability vs acoustic dose (Evans blue model)")

dose_range = np.linspace(0.0, 5.0, 400)  # arbitrary dose units: MI²·s
d50_stable = 1.2   # dose for half-max opening (stable cavitation regime)
d50_inert  = 0.4   # inertial cavitation: faster opening but irreversible risk

# P(D) = D^n / (D₅₀^n + D^n) via kw.bbb_permeability_hill (McDannold 2008)
perm_stable = np.asarray(kw.bbb_permeability_hill(dose_range, d50_stable, 2.5))
perm_inert  = np.asarray(kw.bbb_permeability_hill(dose_range, d50_inert,  1.8))

# Damage threshold: sigmoid beyond ~ 3.5 dose units
damage_thresh = 3.5
damage = 1.0 / (1.0 + np.exp(-4.0 * (dose_range - damage_thresh)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dose_range, perm_stable, color="#4dac26", lw=1.8,
        label="Stable cavitation (reversible opening)")
ax.plot(dose_range, perm_inert, color="#d6604d", lw=1.8, ls="--",
        label="Inertial cavitation (higher opening, damage risk)")
ax.plot(dose_range, damage, color="#a50026", lw=1.4, ls=":",
        label="Tissue damage probability")
ax.axvline(damage_thresh, color="#888", lw=0.8, ls="--", alpha=0.6)
ax.text(damage_thresh + 0.05, 0.5, "Damage\nthreshold", fontsize=8, color="#888")
ax.set_xlabel("Acoustic dose (MI²·s, normalised)")
ax.set_ylabel("Normalised BBB permeability / Damage probability")
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 1.05)
ax.set_title("BBB permeability vs acoustic dose\n"
             "Hill model fit to Evans-blue extravasation (McDannold 2008)")
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
savefig("fig03_bbb_permeability_dose")
plt.close(fig)


# ── Figure 04: Thermal safety — focal temperature rise + CEM43 ───────────────
print("[fig04] LIFU thermal safety: ThermalSimulation (Pennes bioheat, Rust) + CEM43")

# Duck 1990 / Hasgall IT'IS v4.1 — brain tissue
RHO_T  = 1040.0     # kg/m³  tissue density
CP_T   = 3600.0     # J/(kg·K)  specific heat
K_T    = 0.50       # W/(m·K)  thermal conductivity
ALPHA_T = 3.5       # Np/m  absorption at 1 MHz
WB     = 0.01       # s⁻¹   blood perfusion rate (brain)
RHO_B  = 1060.0     # kg/m³  blood density
CP_B   = 3600.0     # J/(kg·K)  blood specific heat
T_ART  = 37.0       # °C    arterial blood temperature

# LIFU pulse sequence
ISATA  = 10.0       # W/cm²  spatial-average temporal-average intensity
DC     = 0.05       # 5% duty cycle (50 ms on, 950 ms off per second)
ISPTP  = ISATA / DC # W/cm²  spatial-peak temporal-peak

t_max  = 120.0      # s  total sonication duration
DT_TH  = 0.01       # s  Pennes bioheat time step
N_TH   = int(t_max / DT_TH)  # 12 000 steps

# Diffusivity check: D = K/(ρ·cp) = 0.5/(1040·3600) ≈ 1.34×10⁻⁷ m²/s
# For DX=1mm: dt_max_1D = dx²/(2D) = 1e-6/(2·1.34e-7) ≈ 3.7 s >> DT_TH ✓

# Focal heat source: time-averaged Q_eff = 2·α·ISATA [W/m³]
# (thermal time constant τ = ρcp/(WB·ρB·cpB) ≈ 98 s >> 1 s pulse period →
#  pulsed Q is thermally equivalent to its duty-cycle average on this timescale)
Q_EFF = 2.0 * ALPHA_T * ISATA * 1e4  # W/m³  (ISATA W/cm² × 1e4 → W/m²)

# 1-D tissue column (200 mm, 1 mm spacing); focal point at centre (ix=100)
NX_TH = 200
DX_TH = 1e-3        # m
IX_FOC = NX_TH // 2

Q_field = np.zeros((NX_TH, 1, 1))
Q_field[IX_FOC, 0, 0] = Q_EFF

sensor_mask = np.zeros((NX_TH, 1, 1), dtype=bool)
sensor_mask[IX_FOC, 0, 0] = True

print(f"  Q_eff = {Q_EFF:.0f} W/m³  |  N_steps = {N_TH}  |  t_max = {t_max:.0f} s")
sim_th = kw.ThermalSimulation(
    NX_TH, 1, 1, DX_TH, DX_TH, DX_TH,
    thermal_conductivity=K_T,
    density=RHO_T,
    specific_heat=CP_T,
    enable_bioheat=True,
    perfusion_rate=WB,
    blood_density=RHO_B,
    blood_specific_heat=CP_B,
    arterial_temperature=T_ART,
    initial_temperature=T_ART,
    track_thermal_dose=True,
)
res_th = sim_th.run(N_TH, DT_TH, heat_source=Q_field, sensor_mask=sensor_mask)
T_arr = np.asarray(res_th.temperature_at_sensors)[0]   # shape (N_TH,)
t_arr = np.asarray(res_th.time)                         # shape (N_TH,)

# Cumulative CEM43 via kw.compute_cem43 on growing sub-trajectories
# Downsampled to 200 query points (each query is O(i) → total O(N_TH·n_q/2))
n_cem_query = 200
query_indices = np.linspace(0, N_TH - 1, n_cem_query, dtype=int)
cem43_sparse = np.array([
    kw.compute_cem43(T_arr[:idx + 1].astype(float), DT_TH)
    for idx in query_indices
])
# Interpolate sparse samples to full time axis
cem43_arr = np.interp(np.arange(N_TH), query_indices, cem43_sparse)
T_max_C = float(T_arr.max())
print(f"  T_peak = {T_max_C:.2f} °C  |  CEM43_total = {cem43_sparse[-1]:.4g} min")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(t_arr, T_arr, color="#d6604d", lw=1.2)
ax1.axhline(T_ART, color="#888", lw=0.8, ls="--", label="$T_{art}$ = 37°C")
ax1.axhline(43.0, color="#a50026", lw=0.8, ls=":", label="43°C safety threshold")
ax1.set_ylabel("Focal temperature (°C)")
ax1.set_title(
    f"LIFU thermal safety — ISATA = {ISATA:.0f} W/cm², DC = {DC*100:.0f}%\n"
    "Pennes bioheat (pykwavers ThermalSimulation); time-averaged Q"
)
ax1.legend(loc="lower right")

ax2.semilogy(t_arr, cem43_arr + 1e-10, color="#2166ac", lw=1.2)
ax2.axhline(0.25, color="#a50026", lw=0.8, ls=":",
            label="CEM43 ≥ 0.25 min: probable damage threshold (brain)")
ax2.set_ylabel("CEM43 (min, log scale)")
ax2.set_xlabel("Sonication time (s)")
ax2.legend(loc="lower right")
ax2.set_xlim(0, t_max)

fig.tight_layout()
savefig("fig04_lifu_thermal_safety")
plt.close(fig)


# ── Figure 05: CEUS contrast enhancement vs MB concentration ─────────────────
print("[fig05] CEUS signal vs microbubble concentration")

c_range = np.linspace(0.0, 50.0, 400)  # µL(gas)/mL
# I_bs = σ_bs·N_V·exp(−2·σ_ext·N_V·thickness) via kw.ceus_backscatter_signal
# (de Jong 1991; single-scatter + MB-layer self-attenuation)
signal = np.asarray(kw.ceus_backscatter_signal(c_range, 2.5e-8, 10e-3))
signal_db = 20.0 * np.log10(signal / np.max(signal) + 1e-12)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(c_range, signal_db, color="#2166ac", lw=1.6)
ax.axvline(c_range[np.argmax(signal)], color="#4dac26", lw=1.0, ls="--",
           label=f"Optimal concentration ≈ {c_range[np.argmax(signal)]:.1f} µL/mL")
ax.set_xlabel("MB gas concentration (µL gas / mL tissue)")
ax.set_ylabel("CEUS backscatter signal (dB re peak)")
ax.set_title("CEUS signal vs microbubble concentration\n"
             "Single-scattering + attenuation model; peak defines working range")
ax.set_xlim(0, 50)
ax.legend()
fig.tight_layout()
savefig("fig05_ceus_signal_vs_concentration")
plt.close(fig)


# ── Figure 06: BBB opening window — permeability vs time post-sonication ──────
print("[fig06] BBB opening window: permeability vs time post-sonication")

t_h = np.linspace(0.0, 48.0, 800)  # hours post-sonication

# Three parameter regimes (McDannold 2008 / Tung 2010)
scenarios = [
    (1.0, 0.85, "#2166ac", "Low dose (MI=0.3, stable SC)"),
    (2.0, 0.92, "#4dac26", "Moderate dose (MI=0.45, stable SC)"),
    (5.0, 0.98, "#d6604d", "High dose (MI=0.65, mixed SC/IC)"),
]

fig, ax = plt.subplots(figsize=(8, 4))
for tau_close, perm_pk, col, lbl in scenarios:
    # P(t) = perm_peak·[0.6·exp(-t/τ_fast)+0.4·exp(-t/τ_slow)]
    # via kw.bbb_closure_kinetics (Deffieux & Konofagou 2010 §IV)
    p = np.asarray(kw.bbb_closure_kinetics(t_h, tau_close, perm_pk))
    ax.plot(t_h, p, color=col, lw=1.6, label=lbl)

# Baseline permeability
ax.axhline(0.0, color="#bbb", lw=0.6, ls="--")
ax.axvline(24.0, color="#888", lw=0.8, ls=":", alpha=0.6, label="24 h landmark")
ax.set_xlabel("Time post-sonication (hours)")
ax.set_ylabel("Normalised BBB permeability")
ax.set_xlim(0, 48)
ax.set_ylim(-0.05, 1.1)
ax.set_title("BBB opening window — bi-exponential closure kinetics\n"
             "Drug delivery window: first 6–24 h post-sonication")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
savefig("fig06_bbb_opening_window")
plt.close(fig)


# ── Figure 07: Multi-spot ray paths and delay laws ───────────────────────────
# Multi-target BBB opening (and multi-focus histotripsy) treat several focal
# sub-spots in one shot.  All array geometry, ray-path/delay-to-target laws and
# the synthesised simultaneous multi-focus field are computed in the kwavers
# Rust core (linear_array_positions, multi_focus_delay_laws_2d,
# multi_focus_field_magnitude_2d); Python only renders.
print("[fig07] Multi-spot ray paths + delay laws (6 sub-spots, Rust core)")

C_BRAIN = 1500.0            # m/s   brain sound speed
F_BBB = 0.5e6              # Hz    transcranial LIFU centre frequency
LAM_BBB = C_BRAIN / F_BBB  # m     wavelength (3 mm)
N_EL = 48                  # phased-array elements (aperture ≈ 72 mm, F# ≈ 0.8)
PITCH = LAM_BBB / 2.0      # m     half-wavelength pitch (grating-lobe free)
EX, EZ = kw.linear_array_positions(N_EL, PITCH)
EX = np.asarray(EX)
EZ = np.asarray(EZ)

# Six sub-spots: a hexagonal cluster around a target 60 mm deep, ~6 mm spacing
# (a representative multi-target BBB-opening montage).
TARGET_Z = 60.0e-3
RING_R = 6.0e-3
spot_x = np.array([0.0,
                   RING_R, RING_R * 0.5, -RING_R * 0.5,
                   -RING_R, RING_R * 0.5]) + 0.0
spot_z = TARGET_Z + np.array([0.0,
                              0.0, RING_R * 0.866, RING_R * 0.866,
                              0.0, -RING_R * 0.866])
N_SPOT = spot_x.size

# Per-element delay laws to each sub-spot (n_spots × n_elem) — Rust.
delays = np.asarray(
    kw.multi_focus_delay_laws_2d(EX, EZ, spot_x, spot_z, C_BRAIN)
)  # [s]

# Synthesised simultaneous multi-focus field via phase-conjugation — Rust.
x_arr = np.linspace(-0.018, 0.018, 281)
z_arr = np.linspace(0.030, 0.085, 281)
amp = np.ones(N_SPOT)
field = np.asarray(
    kw.multi_focus_field_magnitude_2d(
        x_arr, z_arr, EX, EZ, spot_x, spot_z, amp, F_BBB, C_BRAIN
    )
)  # (NX, NZ), normalised to peak

spot_colors = ["#d6604d", "#2166ac", "#4dac26", "#9970ab", "#e08214", "#01665e"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# Panel A: ray paths from a sparse set of elements to every sub-spot.
axA = axes[0]
elem_show = np.linspace(0, N_EL - 1, 9).astype(int)
for j in range(N_SPOT):
    col = spot_colors[j]
    for ie in elem_show:
        axA.plot([EX[ie] * 1e3, spot_x[j] * 1e3],
                 [EZ[ie] * 1e3, spot_z[j] * 1e3],
                 color=col, lw=0.5, alpha=0.45)
    axA.plot(spot_x[j] * 1e3, spot_z[j] * 1e3, "o", color=col,
             ms=9, mec="k", mew=0.6, label=f"spot {j + 1}", zorder=5)
axA.plot(EX * 1e3, EZ * 1e3, "-", color="#333", lw=3, solid_capstyle="butt")
axA.plot(EX[elem_show] * 1e3, EZ[elem_show] * 1e3, "s", color="#333", ms=4)
axA.set_xlabel("Lateral x (mm)")
axA.set_ylabel("Axial z (mm)")
axA.set_title("(A) Ray paths to 6 sub-spots\n"
              f"{N_EL}-element array @ {F_BBB / 1e6:.1f} MHz")
axA.set_aspect("equal")
axA.invert_yaxis()
axA.legend(loc="lower right", fontsize=7, ncol=2)

# Panel B: per-element delay laws (one curve per sub-spot).
axB = axes[1]
elem_idx = np.arange(N_EL)
for j in range(N_SPOT):
    axB.plot(elem_idx, delays[j] * 1e6, color=spot_colors[j],
             lw=1.4, label=f"spot {j + 1}")
axB.set_xlabel("Element index")
axB.set_ylabel("Transmit delay τ (µs)")
axB.set_title("(B) Delay-to-target laws\n"
              "τ = (max r − r) / c per sub-spot")
axB.set_xlim(0, N_EL - 1)
axB.legend(loc="upper center", fontsize=7, ncol=3)

# Panel C: synthesised simultaneous multi-focus field.
axC = axes[2]
field_db = 20.0 * np.log10(field.T + 1e-3)  # transpose → (NZ, NX) for imshow
im = axC.imshow(field_db, origin="lower", aspect="auto", cmap="inferno",
                vmin=-30, vmax=0,
                extent=[x_arr[0] * 1e3, x_arr[-1] * 1e3,
                        z_arr[0] * 1e3, z_arr[-1] * 1e3])
axC.plot(spot_x * 1e3, spot_z * 1e3, "+", color="cyan", ms=11, mew=1.6)
axC.set_xlabel("Lateral x (mm)")
axC.set_ylabel("Axial z (mm)")
axC.set_title("(C) Simultaneous multi-focus field\n"
              "phase-conjugation synthesis (dB re peak)")
cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
cb.set_label("|p| (dB re peak)")

fig.tight_layout()
savefig("fig07_multispot_ray_paths_delays")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# fig08 — Sparse (aperiodic) array widens the treatment envelope
# ─────────────────────────────────────────────────────────────────────────────
# Insightec's 220 kHz transducer enlarges the electronically steerable
# treatment envelope NOT by lowering the frequency but by *sparse, aperiodic*
# element placement.  At one fixed frequency, the same aperture and the same
# element count, a coarse-pitch *periodic* ("dense") layout develops a coherent
# grating lobe once the beam is steered past a pitch-set angle: a second, unin-
# tended focus that can deposit energy off-target.  Re-laying the identical
# number of elements on an *aperiodic* ("sparse") grid scatters that grating
# lobe into a low pedestal, so the beam can be steered much further before any
# secondary lobe rises to the −6 dB safety limit.  The treatment envelope is
# quantified by the grating-lobe-safe steering half-angle: the largest steering
# excursion for which the peak side/grating lobe stays below −6 dB (50 %) of the
# main focus.  All geometry, the steered far-field beam patterns, the grating-
# lobe ratio, and the safe half-angle are computed in the kwavers Rust core
# (linear_array_positions, linear_array_aperiodic_positions,
# steered_beam_pattern_1d, steering_grating_lobe_ratio_1d,
# safe_steering_halfangle); Python only renders.
print("[fig08] Sparse aperiodic array widens the treatment envelope (Rust core)")

C_TC = 1500.0                  # m/s   transcranial sound speed
F0 = 220.0e3                   # Hz    single fixed drive frequency
LAMBDA = C_TC / F0             # m     wavelength (~6.8 mm)
K = 2.0 * np.pi * F0 / C_TC    # rad/m wavenumber
APERTURE = 270.0e-3            # m     full aperture (shared)
N_EL = 48                      # elements (shared between dense and sparse)
A_ELEM = 0.8e-3                # m     element radius (small → broad directivity)
KA = K * A_ELEM                # element directivity parameter

# Dense = coarse-pitch PERIODIC layout; sparse = APERIODIC (golden-ratio dither)
# layout — identical aperture, identical element count, identical frequency.
PITCH = APERTURE / (N_EL - 1)
x_dense, _ = (np.asarray(a) for a in kw.linear_array_positions(N_EL, PITCH))
x_sparse = np.asarray(kw.linear_array_aperiodic_positions(N_EL, APERTURE, 1.0))

# Observation and steering grids (from broadside).
obs = np.linspace(-np.pi / 2, np.pi / 2, 1801)
steer = np.radians(np.linspace(-60.0, 60.0, 241))
HALFWIDTH = np.radians(5.0)    # main-lobe exclusion half-width

# Grating-lobe ratio vs steering angle, and the −6 dB safe steering half-angle.
glr_dense = np.asarray(
    kw.steering_grating_lobe_ratio_1d(x_dense, steer, obs, K, KA, HALFWIDTH)
)
glr_sparse = np.asarray(
    kw.steering_grating_lobe_ratio_1d(x_sparse, steer, obs, K, KA, HALFWIDTH)
)
ha_dense = kw.safe_steering_halfangle(steer, glr_dense, 0.5)
ha_sparse = kw.safe_steering_halfangle(steer, glr_sparse, 0.5)
ratio = ha_sparse / ha_dense if ha_dense > 0 else float("inf")

# A single off-axis steer in the grating-lobe regime, for the beam-pattern cut.
steer_demo = np.radians(30.0)
pat_dense = np.asarray(
    kw.steered_beam_pattern_1d(x_dense, obs, K, steer_demo, KA)
)
pat_sparse = np.asarray(
    kw.steered_beam_pattern_1d(x_sparse, obs, K, steer_demo, KA)
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# Panel A: dense (periodic) vs sparse (aperiodic) element layout.
axA = axes[0]
axA.plot(x_dense * 1e3, np.full_like(x_dense, 1.0), "o", color="#d6604d",
         ms=6, mec="k", mew=0.4, label=f"dense (periodic) {N_EL} el")
axA.plot(x_sparse * 1e3, np.full_like(x_sparse, 0.0), "o", color="#2166ac",
         ms=6, mec="k", mew=0.4, label=f"sparse (aperiodic) {N_EL} el")
axA.set_ylim(-0.6, 1.6)
axA.set_yticks([0.0, 1.0])
axA.set_yticklabels(["sparse", "dense"])
axA.set_xlabel("Element position x (mm)")
axA.set_title(f"(A) Same {N_EL} elements, same {APERTURE * 1e3:.0f} mm aperture\n"
              f"pitch / lambda = {PITCH / LAMBDA:.2f}  @  {F0 / 1e3:.0f} kHz")
axA.legend(loc="upper center", fontsize=7)

# Panel B: steered far-field beam patterns at the off-axis demo steer.
axB = axes[1]
od = np.degrees(obs)
axB.plot(od, 20.0 * np.log10(np.clip(pat_dense, 1e-4, None)),
         color="#d6604d", lw=1.6, label="dense (periodic)")
axB.plot(od, 20.0 * np.log10(np.clip(pat_sparse, 1e-4, None)),
         color="#2166ac", lw=1.6, label="sparse (aperiodic)")
axB.axvline(np.degrees(steer_demo), color="k", ls="--", lw=1.0)
axB.text(np.degrees(steer_demo) + 1.5, -2.0, "steer", fontsize=7)
axB.axhline(-6.0, color="#555555", ls=":", lw=1.0)
axB.text(-88, -5.0, "-6 dB", fontsize=7, va="bottom")
axB.set_xlabel("Observation angle (deg)")
axB.set_ylabel("Beam pattern (dB)")
axB.set_ylim(-40, 2)
axB.set_xlim(-90, 90)
axB.set_title(f"(B) Beam pattern steered to {np.degrees(steer_demo):.0f} deg\n"
              "dense grating lobe vs sparse pedestal")
axB.legend(loc="lower center", fontsize=7)

# Panel C: grating-lobe ratio vs steering angle, with safe half-angles.
axC = axes[2]
sd = np.degrees(steer)
axC.plot(sd, glr_dense, color="#d6604d", lw=2.0, label="dense (periodic)")
axC.plot(sd, glr_sparse, color="#2166ac", lw=2.0, label="sparse (aperiodic)")
axC.axhline(0.5, color="k", ls="--", lw=1.0)
axC.text(-58, 0.52, "-6 dB grating-lobe limit", fontsize=7, va="bottom")
for ha, col in ((ha_dense, "#d6604d"), (ha_sparse, "#2166ac")):
    axC.axvline(np.degrees(ha), color=col, ls=":", lw=1.4)
    axC.axvline(-np.degrees(ha), color=col, ls=":", lw=1.4)
axC.annotate("", xy=(np.degrees(ha_sparse), 0.18),
             xytext=(-np.degrees(ha_sparse), 0.18),
             arrowprops=dict(arrowstyle="<->", color="#2166ac", lw=1.4))
axC.text(0.0, 0.20, f"+/-{np.degrees(ha_sparse):.0f} deg", color="#2166ac",
         fontsize=8, ha="center", va="bottom")
axC.annotate("", xy=(np.degrees(ha_dense), 0.08),
             xytext=(-np.degrees(ha_dense), 0.08),
             arrowprops=dict(arrowstyle="<->", color="#d6604d", lw=1.4))
axC.text(0.0, 0.10, f"+/-{np.degrees(ha_dense):.0f} deg", color="#d6604d",
         fontsize=8, ha="center", va="bottom")
axC.set_xlabel("Steering angle (deg)")
axC.set_ylabel("grating-lobe ratio G")
axC.set_ylim(0.0, 1.05)
axC.set_xlim(-60, 60)
axC.set_title("(C) Grating-lobe-safe steering envelope\n"
              f"sparse / dense half-angle = {ratio:.2f}x")
axC.legend(loc="upper right", fontsize=8)

fig.tight_layout()
savefig("fig08_sparse_treatment_envelope")
plt.close(fig)
print(f"  -6 dB safe steering half-angle: dense={np.degrees(ha_dense):.1f} deg, "
      f"sparse={np.degrees(ha_sparse):.1f} deg, ratio={ratio:.2f}x")

print("[ch24] All figures complete.")
