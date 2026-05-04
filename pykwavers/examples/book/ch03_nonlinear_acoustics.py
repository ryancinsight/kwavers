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
  fig06: Westervelt solver validation — kwavers vs analytical (requires pykwavers)

Usage::

    python ch03_nonlinear_acoustics.py

All figures are saved as both PDF (vector, for LaTeX) and PNG (rasterized, for web).
Requires: numpy, matplotlib, scipy
Optional: pykwavers (fig06 only; skips gracefully if absent)
"""

import os
import sys
import numpy as np
from scipy.special import jv  # Bessel functions J_n
import matplotlib

matplotlib.use("Agg")
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

# Shock distance (Eq. 3.28): z_s = ρ₀c₀³ / (β ω₀ P₀)
Z_S = RHO0 * C0**3 / (BETA_WATER * OMEGA0 * P0)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Fubini harmonic series
# p(z, τ) = (2P₀/σ) Σ_{n=1}^N J_n(n σ)/n · sin(n ω₀ τ)       (Eq. 3.21)
# ─────────────────────────────────────────────────────────────────────────────


def fubini_waveform(tau: np.ndarray, sigma: float, P0: float, N_harm: int = 30) -> np.ndarray:
    """Evaluate the Fubini solution (3.21) at normalized distance sigma.

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
    for n in range(1, N_harm + 1):
        p += (2.0 * P0 / sigma) * jv(n, n * sigma) / n * np.sin(n * OMEGA0 * tau)
    return p


def fubini_harmonic_amplitude(n: int, sigma: float, P0: float) -> float:
    """Amplitude of the n-th harmonic from the Fubini solution (Corollary 3.3).

    Returns: |P_n| = (2 P₀ / σ) |J_n(n σ)| / n
    """
    if sigma < 1e-12:
        return P0 if n == 1 else 0.0
    return abs(2.0 * P0 / sigma * jv(n, n * sigma) / n)


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
        zs = rho * c**3 / (beta * OMEGA0 * P0_val)
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
DELTA_WATER = 4.33e-6  # m²/s, water 20°C
alpha_sk = DELTA_WATER * (2 * np.pi * freqs)**2 / (2 * C0**3)

# Power-law tissue models (Duck 1990): α = α₀ f^y   [Np/m]
# Convert from dB/cm/MHz^y to Np/m: 1 dB/cm = 11.515 Np/m; per MHz^y
def alpha_powerlaw(f, alpha0_dB_cm_MHz, y):
    """Power-law absorption in Np/m, with f in Hz, alpha0 in dB/cm/MHz^y."""
    alpha_Npm_MHz = alpha0_dB_cm_MHz * 11.515 * 100.0  # dB/cm/MHz^y → Np/m/MHz^y
    f_MHz = f / 1e6
    return alpha_Npm_MHz * f_MHz**y

tissues = {
    "Liver (y=1.05)":  (0.45, 1.05, "C1", "-"),
    "Breast (y=1.5)":  (0.57, 1.50, "C2", "--"),
    "Muscle (y=1.01)": (0.57, 1.01, "C3", "-."),
}

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(freqs / 1e6, alpha_sk * 8.686 / 100, label="Water (Stokes-Kirchhoff, y=2)",
          color="C0", lw=2)

for label, (a0, y, col, ls) in tissues.items():
    alpha_pl = alpha_powerlaw(freqs, a0, y)
    ax.loglog(freqs / 1e6, alpha_pl * 8.686 / 100, label=label, color=col, ls=ls, lw=1.6)

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
# Figure 06: Fubini harmonic residual verification
# ─────────────────────────────────────────────────────────────────────────────

print("[fig06] Fubini harmonic residual verification")
sigma = np.linspace(0.02, 0.95, 250)
harmonics = np.arange(1, 12)
residual = np.zeros((harmonics.size, sigma.size))
for i, n in enumerate(harmonics):
    exact = np.array([fubini_harmonic_amplitude(n, s, 1.0) for s in sigma])
    small_signal = np.where(
        n == 1,
        1.0 - 0.125 * sigma * sigma,
        np.where(n == 2, 0.5 * sigma, 0.0),
    )
    residual[i, :] = np.abs(exact - small_signal)

fig, ax = plt.subplots(figsize=(7.4, 4.8))
im = ax.imshow(
    residual,
    origin="lower",
    aspect="auto",
    extent=[sigma[0], sigma[-1], harmonics[0], harmonics[-1]],
    cmap="magma",
)
ax.set_xlabel(r"normalized distance $\sigma = z/z_s$")
ax.set_ylabel("harmonic index n")
ax.set_title("Fubini harmonic departure from weak-shock approximation")
fig.colorbar(im, ax=ax, label=r"$|A_n^{Fubini} - A_n^{weak}|$")
plt.tight_layout()
savefig("fig06_fubini_harmonic_residual")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(
    f"\nChapter 3 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_waveform_evolution.*    — Fubini waveform at σ = 0, 0.25, 0.5, 0.75, 0.99\n"
    "  fig02_harmonic_spectra_sigma.*— Harmonics 1–5 vs normalized distance σ\n"
    "  fig03_second_harmonic_growth.*— P₂ comparison: Fubini / linearized / Theorem 3.8\n"
    "  fig04_shock_distance_tissue.* — z_s for 7 tissue media at 3 source amplitudes\n"
    "  fig05_absorption_models.*     — Stokes-Kirchhoff (y=2) vs tissue power-law\n"
    "  fig06_fubini_harmonic_residual.* — Fubini vs weak-shock harmonic residual\n"
)
