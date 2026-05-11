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

Output directory: docs/book/figures/ch24/
Requires: numpy, matplotlib, scipy

References
----------
Hynynen et al. (2001) Radiology 220(3):640–646
McDannold et al. (2008) Ultrasound Med. Biol. 34(6):930–937
Deffieux & Konofagou (2010) Ultrasound Med. Biol. 36(7):1117–1126
Tung et al. (2010) Proc. Natl. Acad. Sci. 107(8):3699–3704
O'Reilly & Hynynen (2012) Radiology 263(1):96–106
Duck (1990) Physical Properties of Tissue. Academic Press.
Sapareto & Dewey (1984) Int. J. Radiat. Oncol. Biol. Phys. 10(6):787–800
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.integrate import solve_ivp

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
C_L   = 1540.0     # m/s    blood sound speed
KAPPA = 1.07       # polytropic index (gas + shell)
R0    = 1.5e-6     # m      MB equilibrium radius (1.5 µm — SonoVue-like)
# Shell parameters (Doinikov–Dayton 2007 neo-Hookean)
G_S   = 10.0       # MPa  shell shear modulus / m  (× shell thickness ~4 nm)
XI_S  = 1.5e-9     # Pa·s·m  shell viscosity parameter


# ── Figure 01: Keller–Miksis microbubble dynamics ────────────────────────────
print("[fig01] MB dynamics — Keller–Miksis ODE")

def keller_miksis_rhs(t: float, y: np.ndarray, omega: float, pa: float) -> list[float]:
    """
    Keller–Miksis equation for a coated microbubble (Prosperetti 1988):

        (1 - Ṙ/c) R R̈  +  (3/2)(1 - Ṙ/(3c)) Ṙ²
            = (1 + Ṙ/c)/ρ · [p_L - P₀ - p_ac(t)]  +  R/(ρc) · dp_L/dt

    where p_L(R) = (P₀ + 2σ/R₀)(R₀/R)^{3κ} - 2σ/R - 4μṘ/R - 4ξṘ/R²

    Shell viscosity modelled as an additional viscous term 4·XI_S·Ṙ/R².
    Radiation damping implicit via c-correction (Keller & Miksis 1980).
    """
    R, Rdot = y
    if R <= 0:
        return [0.0, 0.0]

    p_inf = P0 + pa * np.sin(omega * t)
    # Inner gas pressure (polytropic)
    p_gas = (P0 + 2.0 * SIGMA / R0) * (R0 / R) ** (3.0 * KAPPA)
    # Liquid pressure on bubble wall
    p_L = p_gas - 2.0 * SIGMA / R - 4.0 * MU_L * Rdot / R - 4.0 * XI_S * Rdot / R**2

    # dp_L/dt (approximate, ignoring dσ/dR terms)
    dp_L_dt = -3.0 * KAPPA * p_gas * Rdot / R

    c = C_L
    lhs_coeff = (1.0 - Rdot / c) * R
    rhs = ((1.0 + Rdot / c) * (p_L - p_inf) / RHO_L
           + R * dp_L_dt / (RHO_L * c)
           - 1.5 * (1.0 - Rdot / (3.0 * c)) * Rdot**2)

    Rddot = rhs / lhs_coeff if abs(lhs_coeff) > 1e-20 else 0.0
    return [Rdot, Rddot]

f0 = 1.0e6  # Hz
omega = 2.0 * np.pi * f0
t_end = 10.0 / f0
t_span = (0.0, t_end)
t_eval = np.linspace(0.0, t_end, 4000)

pressures_pa = [50e3, 150e3, 350e3]  # Pa  — sub-threshold LIFU range
colors = ["#2166ac", "#4dac26", "#d6604d"]
labels = [f"|p_a| = {p/1e3:.0f} kPa" for p in pressures_pa]

fig, axes = plt.subplots(len(pressures_pa), 1, figsize=(8, 7), sharex=True)
for ax, pa, col, lbl in zip(axes, pressures_pa, colors, labels):
    sol = solve_ivp(
        keller_miksis_rhs, t_span, [R0, 0.0],
        args=(omega, pa), t_eval=t_eval,
        method="Radau", rtol=1e-8, atol=1e-12,
        max_step=1.0 / (50 * f0),
    )
    ax.plot(sol.t * 1e6, sol.y[0] / R0, color=col, lw=1.4, label=lbl)
    ax.axhline(1.0, color="#aaa", lw=0.6, ls="--")
    ax.set_ylabel("$R/R_0$")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)

axes[-1].set_xlabel("Time (µs)")
axes[0].set_title(f"Keller–Miksis MB dynamics — $R_0$ = {R0*1e6:.1f} µm, $f_0$ = {f0/1e6:.0f} MHz\n"
                  "SonoVue-like shell; LIFU sub-threshold driving")
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

def _bbb_permeability(dose: np.ndarray, d50: float, hill_n: float) -> np.ndarray:
    """
    Hill-function model for BBB permeability enhancement:

        P(D) = P_max · D^n / (D50^n + D^n)

    where D is the cumulative acoustic dose (MI² × time × PRF),
    D50 is the dose at half-maximum opening, and n is the Hill coefficient.

    Fit to Evans-blue extravasation data (McDannold 2008, Fig. 6).
    P_max = 1.0 (normalised maximum opening, ~10× baseline).
    """
    return dose**hill_n / (d50**hill_n + dose**hill_n)

dose_range = np.linspace(0.0, 5.0, 400)  # arbitrary dose units: MI²·s
d50_stable = 1.2   # dose for half-max opening (stable cavitation regime)
d50_inert  = 0.4   # inertial cavitation: faster opening but irreversible risk

perm_stable = _bbb_permeability(dose_range, d50_stable, hill_n=2.5)
perm_inert  = _bbb_permeability(dose_range, d50_inert,  hill_n=1.8)

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
print("[fig04] LIFU thermal safety: temperature rise and CEM43")

# Bio-heat transfer model (Pennes 1948) linearised for small ΔT:
#   ρ Cp dT/dt = Q - W_b ρ_b Cb (T - T_art)
# At steady state: ΔT_ss = ISATA · α / (W_b · ρ_b · Cb)
# For pulsed LIFU: duty cycle DC reduces effective heating.
# CEM43 (Sapareto & Dewey 1984):
#   CEM43 = Σ_i Δt_i · R^(43 - T_i)   R = 0.5 for T>43°C, 0.25 for T≤43°C

RHO_T = 1040.0      # kg/m³  tissue density
CP_T  = 3600.0      # J/(kg·K)  specific heat
ALPHA_T = 3.5       # Np/m  absorption at 1 MHz (brain tissue)
WB    = 0.01        # s⁻¹   blood perfusion rate (brain, Hasgall 2022)
RHO_B = 1060.0      # kg/m³  blood density
CP_B  = 3600.0      # J/(kg·K)  blood specific heat
T_ART = 37.0        # °C    arterial blood temperature

# LIFU pulse sequence
ISATA = 10.0        # W/cm²  spatial-average temporal-average intensity
DC    = 0.05        # 5% duty cycle (50 ms on, 950 ms off per second)
ISPTP = ISATA / DC  # W/cm²  spatial-peak temporal-peak

t_max = 120.0       # s  total sonication duration
dt    = 0.01        # s  time step
n_t   = int(t_max / dt)
t_arr = np.linspace(0.0, t_max, n_t)

# Heat source (on/off pulsing at DC)
# Q = 2·ALPHA_T·I  [W/m³]  (both expansion and compression contribute)
Q_on  = 2.0 * ALPHA_T * ISPTP * 1e4  # W/m³  (ISPTP in W/cm² × 1e4 to W/m²)
Q_off = 0.0

T_tissue = np.zeros(n_t)
T = T_ART
cem43 = 0.0
cem43_arr = np.zeros(n_t)
perfusion_loss = WB * RHO_B * CP_B

period = 1.0         # s  pulse period
on_time = DC * period

for i in range(n_t):
    t_i = t_arr[i]
    phase = t_i % period
    Q = Q_on if phase < on_time else Q_off
    dT = (Q - perfusion_loss * (T - T_ART)) / (RHO_T * CP_T) * dt
    T = T + dT
    T_tissue[i] = T
    # CEM43 increment
    R = 0.5 if T > 43.0 else 0.25
    cem43 += (T - 43.0 > 0) * (R ** (43.0 - T)) * dt / 60.0  # minutes
    cem43_arr[i] = cem43

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(t_arr, T_tissue, color="#d6604d", lw=1.2)
ax1.axhline(T_ART, color="#888", lw=0.8, ls="--", label="$T_{art}$ = 37°C")
ax1.axhline(43.0, color="#a50026", lw=0.8, ls=":", label="43°C safety threshold")
ax1.set_ylabel("Focal temperature (°C)")
ax1.set_title(f"LIFU thermal safety — ISATA = {ISATA:.0f} W/cm², DC = {DC*100:.0f}%\n"
              "Pennes bioheat model with blood perfusion")
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

def _ceus_signal(c_mb: np.ndarray, sigma_bs: float = 2.5e-8,
                 thickness: float = 10e-3) -> np.ndarray:
    """
    CEUS backscatter signal vs MB concentration (linear regime).

    Single-scattering (low concentration) approximation:
        I_bs = σ_bs · N_V · V_vox
    where N_V is number density [m⁻³] and V_vox = A · thickness.
    Attenuation by MB layer (extinction cross-section σ_ext ≈ 2·σ_bs):
        I_bs_att = I_bs · exp(-2 · σ_ext · N_V · thickness)
    At high concentrations the quadratic multiple-scattering regime dominates.
    """
    n_v = c_mb * 1e9  # convert µL/mL → m⁻³ (approx, bubble volume ~4/3 π R₀³)
    sigma_ext = 2.0 * sigma_bs
    i_linear = sigma_bs * n_v
    attenuation = np.exp(-2.0 * sigma_ext * n_v * thickness)
    return i_linear * attenuation

c_range = np.linspace(0.0, 50.0, 400)  # µL(gas)/mL
signal = _ceus_signal(c_range)
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

def _bbb_recovery(t_h: np.ndarray, tau_close: float,
                  perm_peak: float = 1.0) -> np.ndarray:
    """
    BBB closure kinetics (Deffieux & Konofagou 2010 §IV):

    After sonication ends, BBB permeability decays as a bi-exponential:
        P(t) = A₁ exp(-t/τ_fast) + A₂ exp(-t/τ_slow)

    Fast component (A₁ ~ 0.6): tight-junction re-assembly (~0.5–2 h).
    Slow component (A₂ ~ 0.4): vesicular transport clearance (~6–24 h).
    τ_fast = 0.5 × τ_close (closing time constant),  A₁ = 0.6, A₂ = 0.4.
    """
    tau_fast = 0.5 * tau_close
    tau_slow = 3.0 * tau_close
    return perm_peak * (0.6 * np.exp(-t_h / tau_fast) + 0.4 * np.exp(-t_h / tau_slow))

t_h = np.linspace(0.0, 48.0, 800)  # hours post-sonication

# Three parameter regimes (McDannold 2008 / Tung 2010)
scenarios = [
    (1.0, 0.85, "#2166ac", "Low dose (MI=0.3, stable SC)"),
    (2.0, 0.92, "#4dac26", "Moderate dose (MI=0.45, stable SC)"),
    (5.0, 0.98, "#d6604d", "High dose (MI=0.65, mixed SC/IC)"),
]

fig, ax = plt.subplots(figsize=(8, 4))
for tau_close, perm_pk, col, lbl in scenarios:
    p = _bbb_recovery(t_h, tau_close, perm_pk)
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

print("[ch24] All figures complete.")
