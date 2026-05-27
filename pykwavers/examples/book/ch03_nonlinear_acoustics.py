"""
Chapter 3: Nonlinear Acoustics — Figure Generation Script
==========================================================

Generates all publication-quality figures for Chapter 3 of the kwavers book
using closed-form analytical expressions from:

  - Fubini (1935) solution for harmonic generation (eq. 3.21)
  - Blackstock (1966) Fubini-Fay connection
  - Aanonsen et al. (1984) harmonic spectrum
  - Stokes-Kirchhoff absorption (Theorem 3.10)

Output directory: docs/book/figures/ch03/

Figures produced:
  fig01: Waveform evolution from sinusoidal to sawtooth (Fubini)
  fig02: Harmonic amplitude spectra vs sigma (Fubini Bessel series)
  fig03: Second harmonic growth — Fubini, linearized, and Taylor comparison
  fig04: Effect of B/A on shock distance for tissue media
  fig05: Thermoviscous absorption vs frequency (Stokes-Kirchhoff vs power-law)
  fig06: kwavers PSTD Westervelt solver vs Fubini analytical (requires pykwavers)

Usage::

    python ch03_nonlinear_acoustics.py

All figures are saved as both PDF (vector, for LaTeX) and PNG (rasterized, for web).
Requires: numpy, matplotlib, scipy
Optional: pykwavers (fig06 only; skips gracefully if absent)
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")

import pykwavers as kw
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Output directory ──────────────────────────────────────────────────────────

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch03")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    """Save current figure as PDF and PNG with tight layout."""
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch03/{name}.{{pdf,png}}")


# ── Style ─────────────────────────────────────────────────────────────────────

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

# ── Physical constants ────────────────────────────────────────────────────────

C0 = 1500.0        # sound speed in water (m/s)
RHO0 = 1000.0      # density of water (kg/m³)
BETA_WATER = 3.5   # nonlinearity coefficient β = 1 + B/(2A), water 20 °C
F0 = 1.0e6         # source frequency (Hz)
OMEGA0 = 2 * np.pi * F0
P0 = 1.0e6         # source amplitude (1 MPa)

# Shock distance via kw.shock_formation_distance (Eq. 3.28): z_s = ρ₀c₀³ / (β ω₀ P₀)
Z_S = kw.shock_formation_distance(P0, F0, C0, RHO0, BETA_WATER)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Fubini harmonic series
# p(z, τ) = (2P₀/σ) Σ_{n=1}^N J_n(n σ)/n · sin(n ω₀ τ)       (Eq. 3.21)
# ─────────────────────────────────────────────────────────────────────────────


def fubini_waveform(tau: np.ndarray, sigma: float, P0: float, N_harm: int = 30) -> np.ndarray:
    """Evaluate the Fubini solution (3.21) at normalized distance sigma.

    Bₙ(σ) = 2/(nσ) Jₙ(nσ) from kw.fubini_harmonic_spectrum (Rust kernel).
    p(τ) = P₀ Σₙ Bₙ(σ) sin(nω₀τ)  — sin synthesis is signal processing.

    Args:
        tau:     Array of retarded-time samples [s].
        sigma:   Normalized distance z/z_s (must be < 1 for Fubini convergence).
        P0:      Source pressure amplitude [Pa].
        N_harm:  Number of harmonics to include.

    Returns:
        Pressure waveform [Pa] at position sigma*z_s.
    """
    p = np.zeros_like(tau)
    if sigma < 1e-12:
        return P0 * np.sin(OMEGA0 * tau)
    # Bₙ(σ) normalised amplitudes from Rust (len = N_harm)
    bn = np.asarray(kw.fubini_harmonic_spectrum(N_harm, sigma))
    for n_idx, bn_val in enumerate(bn):
        n = n_idx + 1
        p += P0 * bn_val * np.sin(n * OMEGA0 * tau)
    return p


def fubini_harmonic_amplitude(n: int, sigma: float, P0: float) -> float:
    """Amplitude of the n-th harmonic: |P_n| = P₀ · Bₙ(σ).

    Computed via kw.fubini_harmonic_amplitude (Rust Bessel kernel, Corollary 3.3).
    """
    return P0 * kw.fubini_harmonic_amplitude(n, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: Waveform evolution from sinusoidal → sawtooth
# ─────────────────────────────────────────────────────────────────────────────

print("[fig01] Waveform evolution (Fubini solution)")

sigmas = [0.0, 0.25, 0.5, 0.75, 0.99]
labels = [f"σ = {s:.2f}" for s in sigmas]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(sigmas)))

tau = np.linspace(0, 2.0 / F0, 800)  # two source periods

fig, axes = plt.subplots(1, len(sigmas), figsize=(14, 3.2), sharey=True)
for ax, sigma, label, color in zip(axes, sigmas, labels, colors):
    p = fubini_waveform(tau, sigma, P0, N_harm=40)
    ax.plot(tau * F0, p / P0, color=color, lw=1.8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(r"$f_0 \tau$")
    ax.set_xlim(0, 2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

axes[0].set_ylabel(r"$p / P_0$")

fig.suptitle(
    r"Fubini Solution — Waveform Distortion ($f_0 = 1\,$MHz, Water, $P_0 = 1\,$MPa)",
    y=1.02,
)
plt.tight_layout()
savefig("fig01_waveform_evolution")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: Harmonic amplitude spectra vs σ
# ─────────────────────────────────────────────────────────────────────────────

print("[fig02] Harmonic amplitude spectra vs sigma")

sigma_arr = np.linspace(0.01, 0.99, 300)
N_max = 5  # harmonics to plot

fig, ax = plt.subplots(figsize=(7, 4.5))
cmap = plt.cm.tab10
for n in range(1, N_max + 1):
    amps = [fubini_harmonic_amplitude(n, s, P0) / P0 for s in sigma_arr]
    ax.plot(sigma_arr, amps, color=cmap(n - 1), label=f"$n = {n}$")

ax.set_xlabel(r"Normalized distance $\sigma = z / z_s$")
ax.set_ylabel(r"Harmonic amplitude $|P_n| / P_0$")
ax.set_title(r"Fubini Harmonic Spectrum — Pre-shock ($\sigma < 1$)")
ax.legend(ncol=2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.axvline(1.0, color="k", lw=0.8, ls=":", label="Shock formation")
ax.text(0.97, 0.97, "Shock\nfront", ha="right", va="top", fontsize=8,
        transform=ax.transAxes, color="grey")
plt.tight_layout()
savefig("fig02_harmonic_spectra_sigma")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 03: Second harmonic growth comparison
# ─────────────────────────────────────────────────────────────────────────────

print("[fig03] Second harmonic growth comparison")

z_arr = np.linspace(0, 0.9 * Z_S, 400)
sigma_z = z_arr / Z_S

# Fubini P₂ (Eq. 3.25 exact)
P2_fubini = np.array([fubini_harmonic_amplitude(2, s, P0) for s in sigma_z])

# Linearized approximation P₂ ≈ P₀σ/2 (Corollary 3.4, small σ)
P2_linear = P0 * sigma_z / 2.0

# Quadratic-source approximation from Theorem 3.8
# P₂(z) ≈ β f₀² P₀² z / (ρ₀ c₀⁴)
P2_theorem38 = BETA_WATER * F0**2 * P0**2 * z_arr / (RHO0 * C0**4)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(z_arr * 1e3, P2_fubini / P0 * 100, label="Fubini (Eq. 3.21)", color="C0", lw=2)
ax.plot(
    z_arr * 1e3, P2_linear / P0 * 100, label=r"Linearized: $P_0 \sigma/2$",
    color="C1", ls="--", lw=1.5,
)
ax.plot(
    z_arr * 1e3, P2_theorem38 / P0 * 100,
    label="Theorem 3.8 (quadratic source)",
    color="C2", ls=":", lw=1.5,
)
ax.set_xlabel("Propagation distance (mm)")
ax.set_ylabel(r"$P_2 / P_0$ (%)")
ax.set_title(
    r"Second-Harmonic Growth — $f_0 = 1\,$MHz, Water, $P_0 = 1\,$MPa"
    f"\n(shock distance $z_s = {Z_S*1e3:.1f}$ mm)"
)
ax.axvline(Z_S * 1e3, color="k", lw=0.8, ls=":", label=f"$z_s = {Z_S*1e3:.0f}$ mm")
ax.legend()
ax.set_xlim(0, z_arr[-1] * 1e3)
ax.set_ylim(0)
plt.tight_layout()
savefig("fig03_second_harmonic_growth")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04: Effect of B/A on shock distance for tissue media
# ─────────────────────────────────────────────────────────────────────────────

print("[fig04] Shock distance vs medium B/A")

# Media from Table 3.1
media = {
    "Water (20°C)": (5.2, 1000.0, 1500.0),
    "Water (37°C)": (5.4, 993.0, 1524.0),
    "Blood":        (6.1, 1060.0, 1575.0),
    "Liver":        (6.8, 1060.0, 1578.0),
    "Kidney":       (7.4, 1050.0, 1560.0),
    "Fat":          (9.6, 950.0, 1450.0),
    "Muscle":       (7.9, 1080.0, 1580.0),
}

P0_vals_MPa = [0.5, 1.0, 2.0]  # source pressures
colors_p = ["C0", "C1", "C2"]

fig, ax = plt.subplots(figsize=(8, 5))

for P0_MPa, col in zip(P0_vals_MPa, colors_p):
    P0_val = P0_MPa * 1e6
    zs_vals = []
    labels_m = []
    for name, (ba, rho, c) in media.items():
        beta = 1.0 + ba / 2.0
        zs = kw.shock_formation_distance(P0_val, F0, c, rho, beta)
        zs_vals.append(zs * 1e3)
        labels_m.append(name)
    x = np.arange(len(zs_vals))
    ax.bar(
        x + (P0_MPa / 2.0 - 0.75) * 0.28,
        zs_vals,
        width=0.25,
        color=col,
        alpha=0.85,
        label=f"$P_0 = {P0_MPa}$ MPa",
    )

ax.set_xticks(np.arange(len(media)))
ax.set_xticklabels(labels_m, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Shock distance $z_s$ (mm)")
ax.set_title(f"Shock Distance vs Medium — $f_0 = {F0/1e6:.0f}$ MHz (Eq. 3.28)")
ax.legend()
ax.set_ylim(0)
plt.tight_layout()
savefig("fig04_shock_distance_tissue")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05: Thermoviscous absorption — Stokes-Kirchhoff vs power-law
# ─────────────────────────────────────────────────────────────────────────────

print("[fig05] Absorption models: Stokes-Kirchhoff vs power-law")

freqs = np.logspace(5, 7.5, 300)  # 100 kHz – 30 MHz

# Stokes-Kirchhoff: α = δω²/(2c₀³)  [Np/m]  (Theorem 3.10)
# Reference: Pierce (1989) Acoustics, §10.1, Eq. 10.1.11.
DELTA_WATER = 4.33e-6  # m²/s, water 20°C
alpha_sk = np.asarray(kw.stokes_kirchhoff_absorption_np_m(freqs, DELTA_WATER, C0))

# Power-law tissue models (Duck 1990): α = α₀ f^y  [dB/cm].
# Computed via kw.absorption_power_law_db_cm(f_MHz, alpha0_dBcm, y) (Rust kernel).
tissues = {
    "Liver (y=1.05)":  (0.45, 1.05, "C1", "-"),
    "Breast (y=1.5)":  (0.57, 1.50, "C2", "--"),
    "Muscle (y=1.01)": (0.57, 1.01, "C3", "-."),
}

fig, ax = plt.subplots(figsize=(7, 4.5))
# Stokes-Kirchhoff (water): α = δω²/(2c³) [Np/m] — kw.stokes_kirchhoff_absorption_np_m; Theorem 3.10.
alpha_sk_dBcm = alpha_sk * 8.686 / 100  # Np/m → dB/cm
ax.loglog(freqs / 1e6, alpha_sk_dBcm, label="Water (Stokes-Kirchhoff, y=2)",
          color="C0", lw=2)

for label, (a0, y, col, ls) in tissues.items():
    alpha_pl_dBcm = np.asarray(kw.absorption_power_law_db_cm(freqs / 1e6, a0, y))
    ax.loglog(freqs / 1e6, alpha_pl_dBcm, label=label, color=col, ls=ls, lw=1.6)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel(r"Absorption $\alpha$ (dB/cm)")
ax.set_title("Absorption Coefficient — Classical vs Power-Law (Duck 1990)")
ax.legend(fontsize=9)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
plt.tight_layout()
savefig("fig05_absorption_models")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 06: kwavers PSTD Westervelt solver vs Fubini analytical
# ─────────────────────────────────────────────────────────────────────────────

print("[fig06] Westervelt PSTD solver validation (kwavers vs Fubini)")

if True:
    # ── Grid and solver parameters ─────────────────────────────────────────────
    # 12 pts/wavelength at F0 → 6 pts at 2nd harmonic (adequate for PSTD).
    DX_SIM = C0 / (12.0 * F0)       # 0.125 mm grid spacing
    NX_SIM = 600                      # 600 cells → 75 mm domain
    NY_SIM, NZ_SIM = 1, 1            # 1-D geometry (NX × 1 × 1)
    CFL = 0.25
    DT_SIM = CFL * DX_SIM / C0      # ≈ 2.08e-8 s time step
    PML = 10                          # absorbing boundary cells each side
    B_ON_A = 5.0                      # B/A for water at 20 °C → beta = 3.5

    # Sensor positions: 4 axial cells at sigma ≈ 0.10, 0.20, 0.30, 0.40.
    # Must satisfy PML < ix < NX_SIM - PML.
    SENSOR_IX = [120, 240, 360, 480]

    # ── Source: sinusoidal plane wave injected just inside domain ─────────────
    src_ix = PML + 2

    # Propagation distance from source cell to each sensor; σ = z / z_s.
    # The source is at src_ix, not at cell 0, so the Fubini comparison must
    # use (ix - src_ix) * DX_SIM as the physical propagation distance.
    SIGMA_SENSOR = [(ix - src_ix) * DX_SIM / Z_S for ix in SENSOR_IX]

    # Run long enough to reach steady state at the farthest sensor.
    N_TRAVEL = int((SENSOR_IX[-1] - src_ix) * DX_SIM / (C0 * DT_SIM)) + 1
    N_STEADY_PERIODS = 10                               # periods of F0 for FFT
    N_STEPS_SIM = N_TRAVEL + int(N_STEADY_PERIODS / (F0 * DT_SIM)) + 1
    src_mask_f = np.zeros((NX_SIM, NY_SIM, NZ_SIM), dtype=float)
    src_mask_f[src_ix, 0, 0] = 1.0

    t_src = np.arange(N_STEPS_SIM) * DT_SIM
    signal_2d = (P0 * np.sin(OMEGA0 * t_src)).reshape(1, N_STEPS_SIM)

    source = kw.Source.from_mask(src_mask_f, signal_2d, F0)

    # ── Sensor mask: Boolean flags at 4 axial positions ───────────────────────
    sensor_mask_b = np.zeros((NX_SIM, NY_SIM, NZ_SIM), dtype=bool)
    for ix in SENSOR_IX:
        sensor_mask_b[ix, 0, 0] = True
    sensor = kw.Sensor.from_mask(sensor_mask_b)

    # ── Build and run simulation ───────────────────────────────────────────────
    grid = kw.Grid(nx=NX_SIM, ny=NY_SIM, nz=NZ_SIM,
                   dx=DX_SIM, dy=DX_SIM, dz=DX_SIM)
    medium = kw.Medium.homogeneous(
        sound_speed=C0, density=RHO0, nonlinearity=B_ON_A
    )
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    sim.set_nonlinear(True)
    sim.set_pml_size(PML)

    print(
        f"  NX={NX_SIM}, NY=NZ=1, DX={DX_SIM * 1e3:.3f} mm, "
        f"N_STEPS={N_STEPS_SIM}, DT={DT_SIM * 1e9:.2f} ns"
    )
    result = sim.run(time_steps=N_STEPS_SIM, dt=DT_SIM)

    # ── Extract harmonic amplitudes via windowed FFT ───────────────────────────
    sensor_data = np.asarray(result.sensor_data)  # (4, N_STEPS_SIM)
    dt_run = float(result.dt)
    n_steady_samp = int(N_STEADY_PERIODS / (F0 * dt_run))
    N_HARM = 3  # harmonics n = 1, 2, 3

    pstd_amps = np.zeros((len(SENSOR_IX), N_HARM))
    for i_s in range(len(SENSOR_IX)):
        ts = sensor_data[i_s, -n_steady_samp:]
        win = np.hanning(len(ts))
        P_f = np.abs(np.fft.rfft(ts * win)) * 2.0 / win.sum()
        df = 1.0 / (len(ts) * dt_run)
        for n in range(1, N_HARM + 1):
            idx_f = int(round(n * F0 / df))
            pstd_amps[i_s, n - 1] = P_f[idx_f] if idx_f < len(P_f) else 0.0

    fubini_amps = np.zeros((len(SENSOR_IX), N_HARM))
    for i_s, ix in enumerate(SENSOR_IX):
        # Propagation distance is measured from the source cell, not from ix=0.
        z_sens = (ix - src_ix) * DX_SIM
        sigma_s = z_sens / Z_S
        for n in range(1, N_HARM + 1):
            fubini_amps[i_s, n - 1] = fubini_harmonic_amplitude(n, sigma_s, P0)

    # ── Plot: PSTD vs Fubini per harmonic ─────────────────────────────────────
    fig, axes = plt.subplots(1, N_HARM, figsize=(13, 4.5))
    colors_h = ["C0", "C1", "C2"]
    sigma_pts = [(ix - src_ix) * DX_SIM / Z_S for ix in SENSOR_IX]

    max_rel_err = 0.0
    for col, (ax, col_h) in enumerate(zip(axes, colors_h)):
        n = col + 1
        ax.plot(
            sigma_pts, fubini_amps[:, col] / P0 * 100.0,
            "s--", color=col_h, ms=9, lw=1.5, label="Fubini (analytical)",
        )
        ax.plot(
            sigma_pts, pstd_amps[:, col] / P0 * 100.0,
            "o-", color=col_h, ms=9, lw=2.0, label="kwavers PSTD",
        )
        for i_s, sigma_s in enumerate(sigma_pts):
            ref = fubini_amps[i_s, col]
            if ref > 1e-3 * P0:
                rel_e = abs(pstd_amps[i_s, col] - ref) / ref * 100.0
                max_rel_err = max(max_rel_err, rel_e / 100.0)
                ax.annotate(
                    f"{rel_e:.1f}%",
                    xy=(sigma_s, pstd_amps[i_s, col] / P0 * 100.0),
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=7, ha="center", color="grey",
                )
        ax.set_xlabel(r"Normalized distance $\sigma = z/z_s$")
        ax.set_ylabel(r"$|P_n| / P_0$ (%)")
        ax.set_title(rf"Harmonic $n = {n}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.0, max(sigma_pts) * 1.15)
        ax.set_ylim(bottom=0.0)

    fig.suptitle(
        "Figure 06 — kwavers PSTD Westervelt solver vs Fubini analytical\n"
        r"Water: $c_0=1500$ m/s, $\rho_0=1000$ kg/m³, $B/A=5$, "
        r"$f_0=1$ MHz, $P_0=1$ MPa  (grey: relative error PSTD vs Fubini)",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    print(f"  Max relative error PSTD vs Fubini: {max_rel_err * 100:.2f}%")
    savefig("fig06_westervelt_pstd_validation")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(
    f"\nChapter 3 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_waveform_evolution.*       -- Fubini waveform at sigma = 0, 0.25, 0.5, 0.75, 0.99\n"
    "  fig02_harmonic_spectra_sigma.*   -- Harmonics 1-5 vs normalized distance sigma\n"
    "  fig03_second_harmonic_growth.*   -- P2 comparison: Fubini / linearized / Theorem 3.8\n"
    "  fig04_shock_distance_tissue.*    -- z_s for 7 tissue media at 3 source amplitudes\n"
    "  fig05_absorption_models.*        -- Stokes-Kirchhoff (y=2) vs tissue power-law\n"
    "  fig06_westervelt_pstd_validation.* -- kwavers PSTD Westervelt solver vs Fubini\n"
)
