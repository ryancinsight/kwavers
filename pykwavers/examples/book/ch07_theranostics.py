"""
Chapter 7: Theranostics — Figure Generation Script
===================================================

Figures produced:
  fig01: Keller-Miksis bubble dynamics — R/R₀ vs time for several driving pressures
  fig02: Minnaert resonance frequency vs bubble radius (Theorem 7.2)
  fig03: PCD spectrum — FFT of bubble wall velocity (Keller-Miksis, SC vs IC)
  fig04: PCD-feedback controller convergence — KM-derived SC/IC signals
  fig05: Closed-loop dose accumulation — CEM43 vs time (pykwavers.compute_cem43)

Physics:
  Bubble dynamics: pykwavers.solve_keller_miksis (Keller & Miksis 1980, Rust RK4).
  Thermal dosimetry: pykwavers.compute_cem43 (Sapareto & Dewey 1984).
  Python is used only for FFT, control logic, and plotting.

Output directory: docs/book/figures/ch07/
Requires: numpy, matplotlib, pykwavers
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pykwavers as kw

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch07")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch07/{name}.{{pdf,png}}")


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

# Physical constants — water at 20°C
RHO_L = 998.0        # density [kg/m³]
MU_L = 1.002e-3      # dynamic viscosity [Pa·s]
SIGMA = 0.0725       # surface tension [N/m]
P0 = 101325.0        # ambient pressure [Pa]
KAPPA = 1.4          # polytropic index (adiabatic)
C_L = 1500.0         # sound speed [m/s]
PV = 2338.0          # vapour pressure at 20°C [Pa]
F0_DRV = 1.0e6       # driving frequency [Hz]
OMEGA = 2 * np.pi * F0_DRV

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run Keller-Miksis integrator (Rust RK4)
# ─────────────────────────────────────────────────────────────────────────────


def _km(r0_m: float, p_ac_pa: float, n_cycles: int = 5, n_per: int = 1000):
    """Return (t_s, R_m, Rdot_m_s) arrays from the Keller-Miksis Rust RK4."""
    t_end = n_cycles / F0_DRV
    n_steps = n_cycles * n_per
    t, r, rd = kw.solve_keller_miksis(
        r0_m, 0.0, P0, p_ac_pa, F0_DRV,
        t_end, n_steps, RHO_L, SIGMA, KAPPA, MU_L, PV, C_L, 0.0,
    )
    return np.asarray(t), np.asarray(r), np.asarray(rd)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: Bubble radius vs time — Keller-Miksis for 4 driving pressures
# ─────────────────────────────────────────────────────────────────────────────

print("[fig01] Keller-Miksis bubble dynamics")

R0 = 2e-6          # 2 μm equilibrium radius
N_CYC_FIG1 = 5     # 5 cycles = 5 μs at 1 MHz
N_PER_FIG1 = 6000  # 6000 steps/cycle → dt = 167 ps (resolves violent collapse)

driving_pressures = [10e3, 50e3, 100e3, 200e3]  # Pa
colors_km = ["C0", "C1", "C2", "C3"]

fig, ax = plt.subplots(figsize=(8, 4.5))
for p_a, col in zip(driving_pressures, colors_km):
    t_km, R_km, _ = _km(R0, p_a, N_CYC_FIG1, N_PER_FIG1)
    MI = kw.mechanical_index(float(p_a), float(F0_DRV))  # kw.mechanical_index(p_Pa, f_Hz)
    ax.plot(
        t_km * 1e6, R_km / R0, color=col,
        label=f"$P_a = {p_a/1e3:.0f}$ kPa (MI={MI:.2f})",
    )

ax.axhline(1.0, color="k", lw=0.5, ls="--")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("$R/R_0$")
ax.set_title(
    f"Keller-Miksis Bubble Dynamics (Eq. 7.1 + compressibility correction)\n"
    f"$R_0 = {R0*1e6:.0f}\\,\\mu$m, $f_0 = {F0_DRV/1e6:.0f}$ MHz — pykwavers Rust RK4"
)
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(0, N_CYC_FIG1 / F0_DRV * 1e6)
plt.tight_layout()
savefig("fig01_keller_miksis_dynamics")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: Minnaert resonance frequency vs bubble radius (Theorem 7.2)
# f_min = 1/(2π R₀) · √(3κ P₀/ρ)  — analytical formula
# ─────────────────────────────────────────────────────────────────────────────

print("[fig02] Minnaert resonance frequency")

R0_arr = np.logspace(-7, -4, 300)  # 0.1 μm to 100 μm
# kw.minnaert_resonance_hz: Theorem 7.2 — f₀ = (1/2πR₀)·√(3κP₀/ρ)
f_minnaert = np.array([kw.minnaert_resonance_hz(r0, KAPPA, P0, RHO_L) for r0 in R0_arr])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(R0_arr * 1e6, f_minnaert / 1e6, color="C0", lw=2, label="Minnaert (Eq. 7.3)")

for f_mark, label_m, col_m in [(1, "1 MHz", "C1"), (5, "5 MHz", "C2"), (0.5, "0.5 MHz", "C3")]:
    # Inverse Minnaert: R₀ = (1/2π f)·√(3κP₀/ρ) — algebraic rearrangement, no binding.
    R_mark = (1.0 / (2 * np.pi * f_mark * 1e6)) * np.sqrt(3 * KAPPA * P0 / RHO_L)
    ax.axvline(R_mark * 1e6, color=col_m, lw=1.0, ls=":", alpha=0.7)
    ax.axhline(f_mark, color=col_m, lw=0.8, ls=":")
    ax.text(
        R_mark * 1e6 * 1.1, f_mark * 0.8,
        f"{label_m}\n$R_0={R_mark*1e6:.1f}\\,\\mu$m",
        fontsize=7, color=col_m,
    )

ax.set_xlabel(r"Equilibrium radius $R_0$ ($\mu$m)")
ax.set_ylabel("Resonance frequency (MHz)")
ax.set_title("Minnaert Resonance (Eq. 7.3) — Gas Bubble in Water")
ax.set_xlim(0.1, 100)
ax.set_ylim(0.05, 50)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend()
plt.tight_layout()
savefig("fig02_minnaert_resonance")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 03: PCD spectrum — FFT of Keller-Miksis wall velocity (SC vs IC)
# The far-field radiated pressure p ∝ dṘ/dt (wall acceleration); the power
# spectrum of Ṙ(t) shows the same harmonic/subharmonic pattern observed by
# passive cavitation detectors in experiment.
# SC case : P_a = 50 kPa (MI = 0.050) — stable oscillation + subharmonic
# IC case : P_a = 400 kPa (MI = 0.400) — nonlinear collapse, broadband noise
# R₀ = 3 μm → Minnaert f_res ≈ 1.09 MHz (near-resonance at 1 MHz driving)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig03] PCD spectrum from Keller-Miksis Rdot FFT")

R0_PCD = 3e-6    # 3 μm (near resonance at 1 MHz)
N_CYC_PCD = 50   # 50 cycles → spectral resolution = 20 kHz
N_PER_PCD = 2000 # 2000 steps/cycle → dt = 0.5 ns

p_cases = [(50e3, "Stable Cavitation (SC)\n$P_a = 50$ kPa, MI = 0.050", "C0"),
           (400e3, "Inertial Cavitation (IC)\n$P_a = 400$ kPa, MI = 0.400", "C3")]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (p_a, title_s, col) in zip(axes, p_cases):
    _, _, rdot = _km(R0_PCD, p_a, N_CYC_PCD, N_PER_PCD)
    dt_km = (N_CYC_PCD / F0_DRV) / (N_CYC_PCD * N_PER_PCD)

    # Discard first 20 cycles (transient), use remaining 30 cycles
    n_trans = 20 * N_PER_PCD
    rdot_s = rdot[n_trans:]
    rdot_s = rdot_s - rdot_s.mean()
    window = np.hanning(len(rdot_s))
    spec = np.abs(np.fft.rfft(rdot_s * window)) ** 2
    freq = np.fft.rfftfreq(len(rdot_s), d=dt_km) / 1e6  # MHz

    spec_dB = 10 * np.log10(spec / spec.max() + 1e-12)
    ax.plot(freq, spec_dB, color=col, lw=0.8)

    for n in range(1, 5):
        ax.axvline(n * F0_DRV / 1e6, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.axvline(0.5 * F0_DRV / 1e6, color="purple", lw=0.8, ls=":", alpha=0.8)
    ax.axvline(1.5 * F0_DRV / 1e6, color="purple", lw=0.6, ls=":", alpha=0.6)
    ax.set_title(title_s)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_xlim(0, 5)

axes[0].set_ylabel("Normalised PSD (dB)")
fig.suptitle(
    "Passive Cavitation Detection (PCD) Spectra — Keller-Miksis R₀=3 μm, f₀=1 MHz\n"
    "(Grey dashed = harmonics nf₀; Purple dotted = sub/ultra-harmonics)",
    y=1.02,
)
plt.tight_layout()
savefig("fig03_pcd_spectrum")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04: PCD feedback controller — SC/IC signals from Keller-Miksis
# At each controller step, run KM for 5 cycles and compute the subharmonic
# (SC signal) and broadband (IC signal) power relative to the fundamental.
# ─────────────────────────────────────────────────────────────────────────────

print("[fig04] PCD feedback controller with KM-derived SC/IC signals")

N_PER_CTRL = 1000   # steps per cycle (speed-optimised for 40 controller steps)
N_CYC_CTRL = 5      # cycles per controller evaluation


def _km_signals(p_a_pa: float) -> tuple[float, float]:
    """Return (sc_signal, ic_signal) from KM Rdot FFT for a given pressure."""
    _, _, rdot = _km(R0_PCD, p_a_pa, N_CYC_CTRL, N_PER_CTRL)
    dt_ctrl = (N_CYC_CTRL / F0_DRV) / (N_CYC_CTRL * N_PER_CTRL)
    # Discard first 2 cycles
    rdot_s = rdot[2 * N_PER_CTRL:] - rdot[2 * N_PER_CTRL:].mean()
    window = np.hanning(len(rdot_s))
    spec = np.abs(np.fft.rfft(rdot_s * window)) ** 2
    freq = np.fft.rfftfreq(len(rdot_s), d=dt_ctrl)  # Hz

    fund_mask = (freq > 0.85 * F0_DRV) & (freq < 1.15 * F0_DRV)
    sub_mask = (freq > 0.40 * F0_DRV) & (freq < 0.60 * F0_DRV)
    broad_mask = (freq > 1.5 * F0_DRV) & (freq < 4.5 * F0_DRV)

    fund_pwr = float(spec[fund_mask].sum()) if fund_mask.any() else 1.0
    if fund_pwr < 1e-30:
        fund_pwr = 1e-30
    sc_sig = float(spec[sub_mask].sum()) / fund_pwr if sub_mask.any() else 0.0
    ic_sig = float(spec[broad_mask].sum()) / fund_pwr if broad_mask.any() else 0.0
    return sc_sig, ic_sig


# Controller parameters
N_PULSES = 40
P_INIT = 50e3     # Pa
P_SC = 80e3       # Pa — SC onset (empirical for R0=3 μm at 1 MHz)
P_IC = 350e3      # Pa — IC onset
GAMMA_UP = 1.05
GAMMA_DOWN = 0.80
P_MIN, P_MAX = 10e3, 500e3

P = P_INIT
P_trace = [P / 1e3]   # kPa
SC_trace: list[float] = []
IC_trace: list[float] = []

print("  running KM for each controller pulse …", end="", flush=True)
for pulse in range(N_PULSES - 1):
    sc, ic = _km_signals(P)
    SC_trace.append(sc)
    IC_trace.append(ic)

    # Thresholds: IC > 0.5× broadband signal at P_IC; SC onset when sub-harmonic appears
    if ic > 0.3:
        P = max(P_MIN, GAMMA_DOWN * P)
    elif sc < 0.05:
        P = min(P_MAX, GAMMA_UP * P)
    P_trace.append(P / 1e3)

sc_last, ic_last = _km_signals(P)
SC_trace.append(sc_last)
IC_trace.append(ic_last)
print(" done")

pulses = np.arange(1, N_PULSES + 1)
SC_arr = np.array(SC_trace)
IC_arr = np.array(IC_trace)

# Normalize for display
SC_norm = SC_arr / (SC_arr.max() + 1e-30)
IC_norm = IC_arr / (IC_arr.max() + 1e-30)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(pulses, P_trace, color="C0", lw=1.8, label="Driving pressure")
ax1.axhspan(P_SC / 1e3, P_IC / 1e3, color="green", alpha=0.12, label="Safe SC window")
ax1.axhline(P_SC / 1e3, color="green", lw=0.8, ls="--")
ax1.axhline(P_IC / 1e3, color="red", lw=0.8, ls="--", label="IC threshold")
ax1.set_ylabel("Pressure $P_a$ (kPa)")
ax1.set_title("PCD-Feedback Controller — KM-Derived SC/IC Signals (Algorithm 7.1)")
ax1.legend(fontsize=8)

ax2.plot(pulses, SC_norm, color="C1", lw=1.4, label="Sub-harmonic SC signal")
ax2.plot(pulses, IC_norm, color="C3", lw=1.2, ls="--", label="IC broadband signal")
ax2.axhline(0.05, color="grey", lw=0.7, ls=":")
ax2.set_xlabel("Pulse number")
ax2.set_ylabel("Normalised signal")
ax2.legend(fontsize=8)

plt.tight_layout()
savefig("fig04_pcd_controller_convergence")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05: Closed-loop dose accumulation — CEM43 via pykwavers.compute_cem43
# Temperature histories represent feedback-controlled focal temperatures
# (as measured by MR thermometry in clinical HIFU).  CEM43 integration uses
# the canonical kwavers Rust implementation (Sapareto & Dewey 1984).
# ─────────────────────────────────────────────────────────────────────────────

print("[fig05] Closed-loop CEM43 dose accumulation (pykwavers.compute_cem43)")

N_STEPS = 60
DT_SON = 0.5   # s per sonication step
T_BASE = 37.0
T_TARGET = 60.0
D_ABLATION = 240.0  # CEM43 ablation threshold [min]

steps_idx = np.arange(N_STEPS)

# (a) Fixed power: linear heating to 60°C in 30 steps, maintained thereafter
T_a = np.where(
    steps_idx < 30,
    T_BASE + (T_TARGET - T_BASE) * steps_idx / 30,
    T_TARGET,
)

# (b) Feedback: T overshoots then converges (represents controller action)
rng = np.random.default_rng(42)
T_b = np.where(
    steps_idx < 20,
    T_BASE + (T_TARGET + 10 - T_BASE) * steps_idx / 20,
    T_TARGET + 5 * np.exp(-0.15 * (steps_idx - 20)) + 2.0 * rng.standard_normal(N_STEPS),
)

# (c) Underdrive: T only reaches 56°C
T_c = np.where(
    steps_idx < 40,
    T_BASE + (56 - T_BASE) * steps_idx / 40,
    56.0,
)

# Cumulative CEM43 via pykwavers.compute_cem43 (Rust Sapareto & Dewey kernel)
def _cumulative_cem43(T_arr: np.ndarray, dt_s: float) -> np.ndarray:
    """Return cumulative CEM43 at each step using pykwavers.compute_cem43."""
    out = np.empty(len(T_arr))
    for i in range(len(T_arr)):
        out[i] = kw.compute_cem43(T_arr[: i + 1].astype(float), dt_s)
    return out


D_a = _cumulative_cem43(T_a, DT_SON)
D_b = _cumulative_cem43(T_b, DT_SON)
D_c = _cumulative_cem43(T_c, DT_SON)

t_axis = steps_idx * DT_SON

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(t_axis, T_a, color="C0", label="Fixed power (a)")
ax1.plot(t_axis, T_b, color="C1", lw=1.2, label="Feedback (b)")
ax1.plot(t_axis, T_c, color="C3", ls="--", label="Underdrive (c)")
ax1.axhline(43.0, color="grey", lw=0.7, ls=":", label="43°C")
ax1.axhline(60.0, color="red", lw=0.8, ls="--", label="60°C")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Controlled Focal Temperature — Clinical HIFU\n(MR thermometry representation)")
ax1.legend(fontsize=8)

ax2.semilogy(t_axis, np.maximum(D_a, 1e-4), color="C0", label="Fixed power (a)")
ax2.semilogy(t_axis, np.maximum(D_b, 1e-4), color="C1", lw=1.2, label="Feedback (b)")
ax2.semilogy(t_axis, np.maximum(D_c, 1e-4), color="C3", ls="--", label="Underdrive (c)")
ax2.axhline(D_ABLATION, color="red", lw=1.0, ls="--", label="Ablation 240 min")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("CEM43 (min)")
ax2.set_title("Monotone Dose Accumulation (Theorem 7.6)\npykwavers.compute_cem43")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", ls=":", alpha=0.3)

plt.tight_layout()
savefig("fig05_closed_loop_cem43")
plt.close()

print(
    f"\nChapter 7 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_keller_miksis_dynamics.*   -- R/R0 vs time (Rust RK4, 4 pressures)\n"
    "  fig02_minnaert_resonance.*       -- f_res vs R0 [analytical Minnaert formula]\n"
    "  fig03_pcd_spectrum.*             -- KM Rdot FFT: SC (50 kPa) vs IC (400 kPa)\n"
    "  fig04_pcd_controller_convergence.* -- Controller with KM-derived SC/IC signals\n"
    "  fig05_closed_loop_cem43.*        -- pykwavers.compute_cem43 dose accumulation\n"
)
