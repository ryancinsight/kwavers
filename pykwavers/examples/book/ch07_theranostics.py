"""
Chapter 7: Theranostics — Figure Generation Script
===================================================

Figures produced:
  fig01: Rayleigh-Plesset bubble dynamics — radius vs time for several driving pressures
  fig02: Minnaert resonance frequency vs bubble radius (Theorem 7.2)
  fig03: Bubble scattering spectrum (fundamental, sub-harmonic, broadband)
  fig04: PCD-feedback controller convergence (SI vs IC threshold)
  fig05: Closed-loop dose accumulation — CEM43 vs treatment step (Theorem 7.6)

Output directory: docs/book/figures/ch07/
Requires: numpy, matplotlib, scipy
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# Physical constants
RHO_L = 998.0       # water density [kg/m³]
MU_L = 1.002e-3     # water viscosity [Pa·s]
SIGMA = 0.0725      # surface tension [N/m]
P0 = 101325.0       # ambient pressure [Pa]
KAPPA = 1.4         # polytropic index (adiabatic)
C_L = 1500.0        # sound speed in water [m/s]
F0_DRV = 1.0e6      # driving frequency [Hz]
OMEGA = 2 * np.pi * F0_DRV

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Rayleigh-Plesset ODE (simplified, incompressible)
# R R̈ + 3/2 Ṙ² = (p_g(R) − p∞(t) − 4μṘ/R − 2σ/R) / ρ_l
# ─────────────────────────────────────────────────────────────────────────────


def rp_ode(t, y, R0, p_a, omega):
    """Rayleigh-Plesset ODE: y = [R, Ṙ]."""
    R, Rdot = y
    if R < 1e-12:
        return [Rdot, 0.0]

    p_g0 = P0 + 2 * SIGMA / R0
    p_g = p_g0 * (R0 / R) ** (3 * KAPPA)
    p_inf = P0 + p_a * np.sin(omega * t)
    p_viscous = 4 * MU_L * Rdot / R
    p_surface = 2 * SIGMA / R

    Rddot = (p_g - p_inf - p_viscous - p_surface) / (RHO_L * R) - 1.5 * Rdot**2 / R
    return [Rdot, Rddot]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: Bubble radius vs time for several driving pressures
# ─────────────────────────────────────────────────────────────────────────────

print("[fig01] Rayleigh-Plesset bubble dynamics")

R0 = 2e-6  # 2 μm equilibrium radius
T_END = 5e-6  # 5 μs = 5 cycles at 1 MHz
t_span = (0.0, T_END)
t_eval = np.linspace(0, T_END, 3000)

driving_pressures = [10e3, 50e3, 100e3, 200e3]  # Pa
colors_rp = ["C0", "C1", "C2", "C3"]

fig, ax = plt.subplots(figsize=(8, 4.5))
for p_a, col in zip(driving_pressures, colors_rp):
    sol = solve_ivp(
        rp_ode,
        t_span,
        [R0, 0.0],
        t_eval=t_eval,
        args=(R0, p_a, OMEGA),
        method="RK45",
        max_step=1e-9,
        rtol=1e-6,
        atol=1e-15,
    )
    if sol.success:
        R = sol.y[0]
        MI = p_a / np.sqrt(F0_DRV / 1e6) / 1e6  # MPa/√MHz
        ax.plot(
            sol.t * 1e6,
            R / R0,
            color=col,
            label=f"$P_a = {p_a/1e3:.0f}$ kPa (MI={MI:.2f})",
        )

ax.axhline(1.0, color="k", lw=0.5, ls="--")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("$R/R_0$")
ax.set_title(
    f"Rayleigh-Plesset Bubble Dynamics (Eq. 7.1)\n"
    f"$R_0 = {R0*1e6:.0f}\\,\\mu$m, $f_0 = {F0_DRV/1e6:.0f}$ MHz"
)
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(0, T_END * 1e6)
plt.tight_layout()
savefig("fig01_rayleigh_plesset_dynamics")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: Minnaert resonance frequency vs bubble radius (Theorem 7.2)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig02] Minnaert resonance frequency")

R0_arr = np.logspace(-7, -4, 300)  # 0.1 μm to 100 μm
f_minnaert = (1.0 / (2 * np.pi * R0_arr)) * np.sqrt(3 * KAPPA * P0 / RHO_L)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(R0_arr * 1e6, f_minnaert / 1e6, color="C0", lw=2, label="Minnaert (Eq. 7.3)")

# Mark medical imaging frequencies
for f_mark, label_m, col_m in [(1, "1 MHz", "C1"), (5, "5 MHz", "C2"), (0.5, "0.5 MHz", "C3")]:
    R_mark = (1.0 / (2 * np.pi * f_mark * 1e6)) * np.sqrt(3 * KAPPA * P0 / RHO_L)
    ax.axvline(R_mark * 1e6, color=col_m, lw=1.0, ls=":", alpha=0.7)
    ax.axhline(f_mark, color=col_m, lw=0.8, ls=":")
    ax.text(R_mark * 1e6 * 1.1, f_mark * 0.8, f"{label_m}\n$R_0={R_mark*1e6:.1f}\\,\\mu$m",
            fontsize=7, color=col_m)

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
# Figure 03: Synthetic PCD spectrum — stable vs inertial cavitation
# ─────────────────────────────────────────────────────────────────────────────

print("[fig03] PCD spectrum: SC vs IC")

np.random.seed(12345)
F_DRV = 1.0e6
N_fft = 16384
FS_pcd = 40e6
freq = np.fft.rfftfreq(N_fft, d=1.0 / FS_pcd) / 1e6  # MHz

def make_pcd_spectrum(f_drv_MHz, harmonics, sub_harm_power, broadband_floor, ic_noise):
    """Synthetic PCD power spectrum."""
    spec = broadband_floor * np.ones(len(freq))
    spec += ic_noise * np.random.exponential(1.0, len(freq))  # IC noise
    # Fundamental and harmonics
    for n, power_dB in harmonics:
        sigma_Hz = 0.05  # MHz
        spec += 10 ** (power_dB / 10) * np.exp(-0.5 * ((freq - n * f_drv_MHz) / sigma_Hz) ** 2)
    # Sub-harmonic
    spec += sub_harm_power * np.exp(-0.5 * ((freq - 0.5 * f_drv_MHz) / 0.05) ** 2)
    # Ultra-harmonic
    spec += (sub_harm_power / 3) * np.exp(-0.5 * ((freq - 1.5 * f_drv_MHz) / 0.05) ** 2)
    return spec

# Stable cavitation: strong harmonics + sub-harmonic, low broadband
spec_sc = make_pcd_spectrum(
    F_DRV / 1e6,
    [(1, 60), (2, 45), (3, 30), (4, 15)],
    sub_harm_power=1000,
    broadband_floor=0.1,
    ic_noise=0.05,
)
# Inertial cavitation: stronger broadband + harmonics
spec_ic = make_pcd_spectrum(
    F_DRV / 1e6,
    [(1, 70), (2, 55), (3, 40), (4, 25)],
    sub_harm_power=200,
    broadband_floor=2.0,
    ic_noise=5.0,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, spec, title_s, col in zip(
    axes, [spec_sc, spec_ic], ["Stable Cavitation (SC)", "Inertial Cavitation (IC)"],
    ["C0", "C3"]
):
    spec_dB = 10 * np.log10(spec + 1e-10)
    ax.plot(freq, spec_dB, color=col, lw=0.7)
    # Mark harmonics
    for n in range(1, 5):
        ax.axvline(n * F_DRV / 1e6, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.axvline(0.5 * F_DRV / 1e6, color="purple", lw=0.8, ls=":", alpha=0.8)
    ax.axvline(1.5 * F_DRV / 1e6, color="purple", lw=0.6, ls=":", alpha=0.6)
    ax.set_title(title_s)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_xlim(0, 5)
    ax.set_ylim(-10, 75)

axes[0].set_ylabel("Power spectral density (dB)")
fig.suptitle(
    "Passive Cavitation Detection (PCD) Spectra — Stable vs Inertial Cavitation\n"
    "(Grey dashed = harmonics nf₀; Purple dotted = sub/ultra-harmonics)",
    y=1.02,
)
plt.tight_layout()
savefig("fig03_pcd_spectrum")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04: PCD feedback controller convergence
# ─────────────────────────────────────────────────────────────────────────────

print("[fig04] PCD feedback controller convergence")

# Simulate controller: pressure level vs pulse number
N_PULSES = 40
P_INIT_kPa = 50.0
P_SC_kPa = 100.0   # SC onset
P_IC_kPa = 300.0   # IC onset
GAMMA_UP = 1.05
GAMMA_DOWN = 0.80
P_TARGET_kPa = 0.5 * (P_SC_kPa + P_IC_kPa)  # center of window
P_MIN, P_MAX = 10.0, 500.0

np.random.seed(0)
P = P_INIT_kPa
P_trace = [P]
SC_trace = []
IC_trace = []

for _ in range(N_PULSES - 1):
    # Synthetic SC and IC signals (monotone in P + noise)
    SC = max(0.0, (P - P_SC_kPa) / P_SC_kPa + 0.05 * np.random.randn())
    IC = max(0.0, (P - P_IC_kPa) / P_IC_kPa + 0.02 * np.random.randn())
    SC_trace.append(SC)
    IC_trace.append(IC)

    if IC > 0.05:
        P = max(P_MIN, GAMMA_DOWN * P)
    elif SC < 0.05:
        P = min(P_MAX, GAMMA_UP * P)
    P_trace.append(P)

SC_trace.append(SC_trace[-1])
IC_trace.append(IC_trace[-1])
pulses = np.arange(1, N_PULSES + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(pulses, P_trace, color="C0", lw=1.8, label="Driving pressure")
ax1.axhspan(P_SC_kPa, P_IC_kPa, color="green", alpha=0.12, label="Safe SC window")
ax1.axhline(P_SC_kPa, color="green", lw=0.8, ls="--")
ax1.axhline(P_IC_kPa, color="red", lw=0.8, ls="--", label="IC threshold")
ax1.set_ylabel("Pressure $P_a$ (kPa)")
ax1.set_title("PCD-Feedback Controller Convergence (Algorithm 7.1)")
ax1.legend(fontsize=8)

ax2.plot(pulses, SC_trace, color="C1", lw=1.4, label="SC signal")
ax2.plot(pulses, IC_trace, color="C3", lw=1.2, ls="--", label="IC broadband noise")
ax2.axhline(0.05, color="grey", lw=0.7, ls=":", label="SC/IC thresholds")
ax2.set_xlabel("Pulse number")
ax2.set_ylabel("Normalized signal")
ax2.legend(fontsize=8)

plt.tight_layout()
savefig("fig04_pcd_controller_convergence")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05: Closed-loop dose accumulation (Theorem 7.6 — CEM43 monotone)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig05] Closed-loop CEM43 dose accumulation")

# Simulate three control scenarios:
# (a) Fixed power, (b) Feedback control (ramp then hold), (c) Underestimated
N_STEPS = 60
DT_sonication = 0.5  # s per sonication
T_base = 37.0
T_target = 60.0
D_ablation = 240.0   # CEM43 threshold [min]

def cem43_rate(T_celsius):
    """CEM43 rate [min/s] at temperature T."""
    R = 0.5 if T_celsius >= 43 else 0.25
    return R ** (43.0 - T_celsius) / 60.0  # per second → per minute

# (a) Fixed power: linear heating to 60°C in 30 steps, maintained
T_a = np.where(
    np.arange(N_STEPS) < 30,
    T_base + (T_target - T_base) * np.arange(N_STEPS) / 30,
    T_target
)
D_a = np.cumsum([cem43_rate(T) * DT_sonication for T in T_a])

# (b) Feedback: T overshoots then converges
noise = 2.0 * np.random.randn(N_STEPS)
T_b = np.where(
    np.arange(N_STEPS) < 20,
    T_base + (T_target + 10 - T_base) * np.arange(N_STEPS) / 20,
    T_target + 5 * np.exp(-0.15 * (np.arange(N_STEPS) - 20)) + noise
)
D_b = np.cumsum([cem43_rate(T) * DT_sonication for T in T_b])

# (c) Underdrive: T only reaches 56°C
T_c = np.where(
    np.arange(N_STEPS) < 40,
    T_base + (56 - T_base) * np.arange(N_STEPS) / 40,
    56.0
)
D_c = np.cumsum([cem43_rate(T) * DT_sonication for T in T_c])

steps = np.arange(N_STEPS) * DT_sonication

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(steps, T_a, color="C0", label="Fixed power (a)")
ax1.plot(steps, T_b, color="C1", lw=1.2, label="Feedback (b)")
ax1.plot(steps, T_c, color="C3", ls="--", label="Underdrive (c)")
ax1.axhline(43.0, color="grey", lw=0.7, ls=":", label="43°C")
ax1.axhline(60.0, color="red", lw=0.8, ls="--", label="60°C")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Temperature Evolution")
ax1.legend(fontsize=8)

ax2.semilogy(steps, np.maximum(D_a, 1e-4), color="C0", label="Fixed power (a)")
ax2.semilogy(steps, np.maximum(D_b, 1e-4), color="C1", lw=1.2, label="Feedback (b)")
ax2.semilogy(steps, np.maximum(D_c, 1e-4), color="C3", ls="--", label="Underdrive (c)")
ax2.axhline(D_ablation, color="red", lw=1.0, ls="--", label="Ablation threshold 240 min")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("CEM43 (min)")
ax2.set_title("Monotone Dose Accumulation (Theorem 7.6)")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", ls=":", alpha=0.3)

plt.tight_layout()
savefig("fig05_closed_loop_cem43")
plt.close()

print(
    f"\nChapter 7 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_rayleigh_plesset_dynamics.* — R/R₀ vs time for 4 driving pressures\n"
    "  fig02_minnaert_resonance.*        — Resonance frequency vs bubble radius\n"
    "  fig03_pcd_spectrum.*              — Synthetic SC and IC PCD spectra\n"
    "  fig04_pcd_controller_convergence.*— Pressure feedback convergence\n"
    "  fig05_closed_loop_cem43.*         — Monotone dose accumulation (3 scenarios)\n"
)
