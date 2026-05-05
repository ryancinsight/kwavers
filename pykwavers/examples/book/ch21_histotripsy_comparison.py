"""
Chapter 21: Histotripsy — Classical vs Millisecond-Pulse Comparison
====================================================================

Side-by-side figures comparing classical (intrinsic-threshold, μs-pulse)
histotripsy with millisecond-pulse (boiling) histotripsy.

Figures produced (all saved as PNG and PDF):
  fig01_pulse_waveforms          — focal pressure waveforms, both regimes
  fig02_cavitation_probability   — erf-CDF intrinsic threshold (Theorem 21.1)
  fig03_bioheat_temperature      — focal T(t) per pulse (Theorem 21.2)
  fig04_cem43_accumulation       — Sapareto-Dewey CEM43 per pulse (Theorem 21.3)
  fig05_mechanism_phase_map      — (τ_p, |p⁻|) regime map

Output directory: docs/book/figures/ch21/
Requires: numpy, matplotlib, scipy
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch21/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.4,
})

# ──────────────────────────────────────────────────────────────────────────
# Physical parameters (liver, 1 MHz)  [Khokhlova 2014, Maxwell 2013]
# ──────────────────────────────────────────────────────────────────────────
F0      = 1.0e6           # carrier frequency [Hz]
RHO0    = 1060.0          # density [kg/m^3]
CP      = 3600.0          # specific heat [J/kg/K]
KAPPA   = 0.51            # thermal conductivity [W/m/K]
ALPHA_F = 7.0             # absorption at fundamental, 1 MHz [Np/m]
ALPHA_S = 100.0           # effective absorption at shock spectrum [Np/m]
T0      = 37.0            # baseline tissue temperature [°C]
T_BOIL  = 100.0           # boiling temperature [°C]

# Intrinsic-threshold parameters [Maxwell 2013, Table II — bovine liver, 1 MHz]
P_T     = 28.2e6          # intrinsic threshold mean [Pa]
SIGMA_T = 0.96e6          # intrinsic threshold std  [Pa]

# Classical (intrinsic-threshold) regime
PNP_C   = 28.5e6          # peak negative pressure [Pa]
TAU_C   = 5.0e-6          # pulse duration [s]  (5 cycles at 1 MHz)
PRF_C   = 200.0           # pulse repetition frequency [Hz]

# Millisecond-pulse (boiling) regime
PNP_M   = 17.0e6          # peak negative pressure [Pa]   — below intrinsic threshold
PPP_M   = 80.0e6          # peak positive pressure (shock-formed) [Pa]
TAU_M   = 10.0e-3         # pulse duration [s] (10 ms)
PRF_M   = 1.0             # pulse repetition frequency [Hz]
I_S_M   = 25e7            # focal shock-rich intensity [W/m^2]  (25 kW/cm^2)


# ──────────────────────────────────────────────────────────────────────────
# Figure 21.1 — Pulse waveforms
# ──────────────────────────────────────────────────────────────────────────
print("[fig01] Pulse waveforms (classical vs ms-pulse)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

# Classical: 5-cycle tone burst, sinusoid with PNP = -28.5 MPa
n_cycles_c = TAU_C * F0
t_c = np.linspace(0.0, TAU_C, int(n_cycles_c * 200))
env_c = np.where(
    (t_c < 0.5e-6) | (t_c > TAU_C - 0.5e-6),
    0.5 * (1.0 - np.cos(2 * np.pi * t_c / 1e-6)),
    1.0,
)
p_c = PNP_C * env_c * np.sin(2 * np.pi * F0 * t_c + np.pi / 2)

axes[0].plot(t_c * 1e6, p_c / 1e6, color="C0", lw=1.2)
axes[0].axhline(-P_T / 1e6, color="r", lw=0.8, ls="--",
                label=f"intrinsic threshold $-p_t = -{P_T/1e6:.1f}$ MPa")
axes[0].set_xlabel("time (μs)")
axes[0].set_ylabel("focal pressure (MPa)")
axes[0].set_title(f"Classical: $\\tau_p={TAU_C*1e6:.0f}$ μs, "
                  f"PNP$={-PNP_C/1e6:.1f}$ MPa")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# Millisecond: shock-formed envelope. Carrier sine + asymmetric shock
# (positive peak amplified, negative peak limited by shock formation)
t_m = np.linspace(0.0, TAU_M, 5000)
phase = 2 * np.pi * F0 * t_m
# Asymmetric shock model: positive peak boosted by 3rd-harmonic in-phase,
# negative half-cycle attenuated. Crude analytical surrogate of Khokhlova
# 2014 measured shock waveform.
fund = np.sin(phase)
shock_pos = np.maximum(fund, 0) ** 2  # peaked
shock_neg = -np.minimum(fund, 0)
p_m = PPP_M * shock_pos - PNP_M * shock_neg

# Show only first 50 μs to make shock visible at ms-pulse scale
mask = t_m < 50e-6
axes[1].plot(t_m[mask] * 1e6, p_m[mask] / 1e6, color="C3", lw=1.2)
axes[1].axhline(-P_T / 1e6, color="r", lw=0.8, ls="--",
                label=f"intrinsic threshold $-p_t = -{P_T/1e6:.1f}$ MPa")
axes[1].set_xlabel("time (μs)  —  envelope continues to 10 ms")
axes[1].set_ylabel("focal pressure (MPa)")
axes[1].set_title(f"Millisecond: $\\tau_p={TAU_M*1e3:.0f}$ ms, "
                  f"PNP$={-PNP_M/1e6:.1f}$ MPa, PPP$={PPP_M/1e6:.0f}$ MPa")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig("fig01_pulse_waveforms")
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 21.2 — Intrinsic-threshold cavitation probability (Theorem 21.1)
# ──────────────────────────────────────────────────────────────────────────
print("[fig02] Cavitation probability vs PNP")

p_minus_scan = np.linspace(20e6, 35e6, 400)
p_cav = 0.5 * (1.0 + erf((p_minus_scan - P_T) / (SIGMA_T * np.sqrt(2.0))))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(p_minus_scan / 1e6, p_cav, color="k", lw=1.6,
        label="erf-CDF, Maxwell 2013")
ax.axvline(P_T / 1e6, color="gray", lw=0.8, ls=":",
           label=f"$p_t = {P_T/1e6:.1f}$ MPa")
# Operating points
p_cav_C = 0.5 * (1.0 + erf((PNP_C - P_T) / (SIGMA_T * np.sqrt(2.0))))
p_cav_M = 0.5 * (1.0 + erf((PNP_M - P_T) / (SIGMA_T * np.sqrt(2.0))))
ax.scatter([PNP_C / 1e6], [p_cav_C], s=70, color="C0", zorder=5,
           label=f"classical: |PNP|$={PNP_C/1e6:.1f}$ MPa, "
                 f"$P_\\mathrm{{cav}}={p_cav_C:.3f}$")
ax.scatter([PNP_M / 1e6], [p_cav_M], s=70, color="C3", marker="s", zorder=5,
           label=f"ms-pulse: |PNP|$={PNP_M/1e6:.1f}$ MPa, "
                 f"$P_\\mathrm{{cav}}={p_cav_M:.2e}$")
ax.set_xlabel("|peak negative pressure| (MPa)")
ax.set_ylabel("single-pulse cavitation probability  $P_\\mathrm{cav}$")
ax.set_title("Intrinsic-threshold cavitation (Theorem 21.1)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig02_cavitation_probability")
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 21.3 — Bioheat focal temperature trajectory (Theorem 21.2)
# ──────────────────────────────────────────────────────────────────────────
print("[fig03] Bioheat temperature rise during single pulse")

# Classical heating rate: I_spta during pulse ~ p^2/(2 rho c)
# but pulse is too short to register on bioheat timescale.
I_C = (PNP_C ** 2) / (2.0 * RHO0 * 1540.0)  # cycle-averaged peak [W/m^2]
Q_C = 2.0 * ALPHA_F * I_C
dT_dt_C = Q_C / (RHO0 * CP)

# Millisecond regime: shock-spectrum absorption against intensity I_S_M
Q_M = 2.0 * ALPHA_S * I_S_M
dT_dt_M = Q_M / (RHO0 * CP)

# Time grids — chosen to make both pulses comparable on log axis.
t_pulse_C = np.linspace(0.0, TAU_C, 500)
t_pulse_M = np.linspace(0.0, TAU_M, 5000)
T_C = np.minimum(T0 + dT_dt_C * t_pulse_C, T_BOIL)
T_M = np.minimum(T0 + dT_dt_M * t_pulse_M, T_BOIL)

t_boil = (T_BOIL - T0) * RHO0 * CP / Q_M
print(f"        analytical t_boil (ms-pulse) = {t_boil*1e3:.2f} ms")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

axes[0].plot(t_pulse_C * 1e6, T_C, color="C0", lw=1.6)
axes[0].axhline(43.0, color="orange", lw=0.7, ls=":", label="43 °C (CEM43 ref)")
axes[0].set_xlabel("time (μs)")
axes[0].set_ylabel("focal temperature (°C)")
axes[0].set_title(f"Classical, $\\Delta T_\\mathrm{{end}}="
                  f"{T_C[-1]-T0:.2e}$ °C")
axes[0].legend(loc="best")
axes[0].set_ylim(36.99, 37.05)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_pulse_M * 1e3, T_M, color="C3", lw=1.6)
axes[1].axhline(43.0, color="orange", lw=0.7, ls=":", label="43 °C")
axes[1].axhline(T_BOIL, color="r", lw=0.8, ls="--", label="100 °C")
axes[1].axvline(t_boil * 1e3, color="k", lw=0.6, ls=":",
                label=f"$t_\\mathrm{{boil}}={t_boil*1e3:.2f}$ ms")
axes[1].set_xlabel("time (ms)")
axes[1].set_ylabel("focal temperature (°C)")
axes[1].set_title(f"Millisecond, peak $T={T_M.max():.0f}$ °C")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig("fig03_bioheat_temperature")
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 21.4 — CEM43 accumulation per single pulse (Theorem 21.3)
# ──────────────────────────────────────────────────────────────────────────
print("[fig04] CEM43 accumulation per single pulse")


def cem43(t_min: np.ndarray, T_celsius: np.ndarray) -> np.ndarray:
    """Sapareto-Dewey CEM43 cumulative integral [min]."""
    R = np.where(T_celsius >= 43.0, 0.5, 0.25)
    integrand = R ** (43.0 - T_celsius)
    # cumulative trapezoidal
    return np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                             * np.diff(t_min))])


t_pulse_C_min = t_pulse_C / 60.0
t_pulse_M_min = t_pulse_M / 60.0
cem_C = cem43(t_pulse_C_min, T_C)
cem_M = cem43(t_pulse_M_min, T_M)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

axes[0].plot(t_pulse_C * 1e6, cem_C, color="C0", lw=1.6)
axes[0].set_xlabel("time (μs)")
axes[0].set_ylabel("cumulative CEM43 (min)")
axes[0].set_title(f"Classical, end-of-pulse CEM43$={cem_C[-1]:.2e}$ min")
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(t_pulse_M * 1e3, np.maximum(cem_M, 1e-30), color="C3", lw=1.6)
axes[1].axhline(240.0, color="r", lw=0.7, ls=":", label="ablation threshold (240 min)")
axes[1].set_xlabel("time (ms)")
axes[1].set_ylabel("cumulative CEM43 (min)")
axes[1].set_title(f"Millisecond, end-of-pulse CEM43$={cem_M[-1]:.2e}$ min")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig("fig04_cem43_accumulation")
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 21.5 — Mechanism phase map in (tau_p, |p^-|)
# ──────────────────────────────────────────────────────────────────────────
print("[fig05] Mechanism phase map")

tau_grid  = np.logspace(-7, -1, 300)        # 100 ns to 100 ms
pnp_grid  = np.linspace(0.0, 35.0, 300) * 1e6
TAU, PNP  = np.meshgrid(tau_grid, pnp_grid)

# Region indicators:
#   1: intrinsic-threshold (classical)  — |PNP| > p_t  AND tau_p < 100 us
#   2: shock-scattering  (ms / boiling) — |PNP| < p_t  AND tau_p > t_boil
#   3: thermal ablation (HIFU)          — |PNP| < p_t  AND tau_p > 1 s
region = np.zeros_like(TAU, dtype=int)
region[(PNP >= P_T) & (TAU <= 100e-6)] = 1
region[(PNP <  P_T) & (TAU >= t_boil) & (TAU < 1.0)] = 2
region[(PNP <  P_T) & (TAU >= 1.0)] = 3

fig, ax = plt.subplots(figsize=(8.0, 5.0))
cmap = plt.matplotlib.colors.ListedColormap(
    ["white", "#cfe8ff", "#ffd8c9", "#fff3b0"]
)
ax.pcolormesh(TAU * 1e3, PNP / 1e6, region, cmap=cmap, shading="auto",
              vmin=0, vmax=3)
ax.set_xscale("log")
ax.axhline(P_T / 1e6, color="k", lw=0.9, ls="--")
ax.text(2e-4, P_T / 1e6 + 0.4, "intrinsic threshold $p_t$",
        fontsize=9, color="k")
ax.axvline(t_boil * 1e3, color="k", lw=0.9, ls=":")
ax.text(t_boil * 1e3 * 1.1, 4.0,
        f"$t_\\mathrm{{boil}}\\approx{t_boil*1e3:.1f}$ ms",
        fontsize=9, color="k", rotation=90)

# Operating points
ax.scatter([TAU_C * 1e3], [PNP_C / 1e6], s=80, color="C0",
           edgecolor="k", zorder=5, label="Classical")
ax.scatter([TAU_M * 1e3], [PNP_M / 1e6], s=80, marker="s",
           color="C3", edgecolor="k", zorder=5, label="Millisecond")

# Region annotations
ax.text(2e-3, 32, "Intrinsic-threshold\nhistotripsy", fontsize=10, ha="center")
ax.text(20.0, 13, "Shock-scattering\n(ms / boiling histotripsy)",
        fontsize=10, ha="center")
ax.text(2e3, 5, "HIFU thermal ablation", fontsize=10, ha="center")

ax.set_xlabel("pulse duration $\\tau_p$ (ms, log scale)")
ax.set_ylabel("|peak negative pressure| (MPa)")
ax.set_title("Histotripsy regime phase map")
ax.set_xlim(1e-4, 1e5)
ax.set_ylim(0, 35)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
savefig("fig05_mechanism_phase_map")
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────
print()
print("─" * 60)
print(f"Chapter 21 figures written to: {OUT_DIR}")
print(f"  classical  P_cav  = {p_cav_C:.4f}")
print(f"  ms-pulse   P_cav  = {p_cav_M:.3e}  (single-shot, no shock-scatter)")
print(f"  classical  CEM43  = {cem_C[-1]:.3e} min  per pulse")
print(f"  ms-pulse   CEM43  = {cem_M[-1]:.3e} min  per pulse")
print(f"  ms-pulse   t_boil = {t_boil*1e3:.2f} ms")
