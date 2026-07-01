"""
Chapter 23: Passive Acoustic Mapping (PAM) for Cerebral Cavitation Detection
=============================================================================

Publication-quality figures for docs/book/passive_acoustic_mapping.md.

PAM reconstructs the spatial distribution of acoustic emissions inside tissue
from a passive multi-element receive aperture.  In transcranial focused-
ultrasound (tFUS) therapy of the brain it distinguishes stable (harmonic)
from inertial (broadband) cavitation — the primary safety and efficacy
biomarker for BBB opening and histotripsy.

Figures produced
----------------
fig01  Cavitation emission spectra — stable (harmonic) vs inertial (broadband)
fig02  Passive delay-and-sum (DAS) sensitivity map vs focal depth
fig03  Spatial coherence function and the van Cittert–Zernike theorem
fig04  Eigenspace PAM beamformer: signal vs noise singular values
fig05  Transcranial attenuation and phase aberration effect on PAM SNR
fig06  Stable/inertial discrimination: cavitation dose accumulation

Output directory: docs/book/figures/ch23/
Requires: numpy, matplotlib

References
----------
Coviello et al. (2015) J. Acoust. Soc. Am. 137(5):2573–2585
Salgaonkar et al. (2009) J. Acoust. Soc. Am. 126(6):3071–3083
O'Reilly & Hynynen (2013) Med. Phys. 40(11):110701
Arnal et al. (2017) IEEE Trans. Med. Imaging 36(7):1543
Gyöngy & Coussios (2010) IEEE Trans. Ultrason. Ferroelectr. Freq. Control 57(6):1356
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch23")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch23/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

# ── Physical parameters ──────────────────────────────────────────────────────
C0 = 1500.0          # m/s  soft-tissue sound speed
F0 = 1.0e6           # Hz   fundamental driving frequency
FS = 40.0e6          # Hz   receive sample rate (typical tFUS system)
N_EL = 128           # elements in receive aperture
PITCH = 3.0e-4       # m    element pitch (0.3 mm)
SKULL_ALPHA = 6.0    # dB/(cm·MHz^1.2)  skull attenuation coefficient
SKULL_THICK = 0.7e-2 # m    calvaria thickness


# ── Figure 01: Cavitation emission spectra ────────────────────────────────────
print("[fig01] Cavitation emission spectra (stable vs inertial)")

f_vec = np.linspace(0.1e6, 5.5e6, 4000)
sc_psd = np.asarray(kw.normalized_cavitation_emission_spectrum(f_vec, F0, "stable"))
ic_psd = np.asarray(kw.normalized_cavitation_emission_spectrum(f_vec, F0, "inertial"))
sc_db = 10.0 * np.log10(np.maximum(sc_psd, np.finfo(float).tiny))
ic_db = 10.0 * np.log10(np.maximum(ic_psd, np.finfo(float).tiny))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(f_vec / 1e6, sc_db, color="#2166ac", label="Stable cavitation (SC)", lw=1.5)
ax.plot(f_vec / 1e6, ic_db, color="#d6604d", label="Inertial cavitation (IC)", lw=1.5)
for n in range(1, 6):
    ax.axvline(n * F0 / 1e6, color="#555", lw=0.5, ls="--", alpha=0.5)
ax.axvline(0.5 * F0 / 1e6, color="#888", lw=0.5, ls=":", alpha=0.6, label="$f_0/2$ sub-harmonic")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Normalised PSD (dB)")
ax.set_xlim(0, 5.5)
ax.set_ylim(-55, 5)
ax.legend(loc="upper right")
ax.set_title("Cavitation emission spectra\n"
             "SC: harmonic lines; IC: elevated broadband floor")
fig.tight_layout()
savefig("fig01_cavitation_spectra")
plt.close(fig)


# ── Figure 02: DAS point-spread / sensitivity map ────────────────────────────
print("[fig02] Passive DAS localization map (kw.passive_acoustic_map_das)")

# Synthetic passive RF only: a single cavitation point source at (0, 40 mm)
# radiates a short 3-cycle pulse. Rust computes receive delays, Gaussian
# emission envelope, carrier phase, and 1/r spreading; the delay-and-sum
# localization map is then computed by the kwavers PAM kernel (§22.3).
el_x = (np.arange(N_EL) - (N_EL - 1) / 2.0) * PITCH
sensor_xyz = np.column_stack([el_x, np.zeros(N_EL), np.zeros(N_EL)])
src = np.array([0.0, 0.0, 40e-3])

N_SAMP = 1800
N_CYCLES = 3.0
passive = np.asarray(kw.passive_cavitation_point_source_rf(
    np.ascontiguousarray(sensor_xyz),
    np.ascontiguousarray(src),
    N_SAMP,
    FS,
    C0,
    F0,
    N_CYCLES,
))

x_range = np.linspace(-15e-3, 15e-3, 100)
z_range = np.linspace(10e-3, 70e-3, 120)
xx, zz = np.meshgrid(x_range, z_range)
grid = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])

das_map = np.asarray(kw.passive_acoustic_map_das(
    np.ascontiguousarray(passive),
    np.ascontiguousarray(sensor_xyz),
    np.ascontiguousarray(grid),
    C0, FS, window_size=256, apodization="hamming", coherence_weighting=True,
)).reshape(len(z_range), len(x_range))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(
    10 * np.log10(das_map / das_map.max() + 1e-12),  # intensity map → 10·log10
    extent=[x_range[0] * 1e3, x_range[-1] * 1e3, z_range[-1] * 1e3, z_range[0] * 1e3],
    aspect="auto", cmap="hot", vmin=-40, vmax=0,
)
ax.plot(src[0] * 1e3, src[2] * 1e3, "c+", ms=12, mew=2, label="true source")
cb = fig.colorbar(im, ax=ax)
cb.set_label("DAS energy (dB re peak)")
ax.set_xlabel("Lateral position (mm)")
ax.set_ylabel("Depth (mm)")
ax.set_title(f"Passive DAS localization — {N_EL}-element aperture, pitch {PITCH*1e3:.1f} mm\n"
             f"kw.passive_acoustic_map_das,  $f_0$ = {F0/1e6:.1f} MHz")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
savefig("fig02_das_sensitivity_map")
plt.close(fig)


# ── Figure 03: Spatial coherence and van Cittert–Zernike theorem ──────────────
print("[fig03] Spatial coherence function (van Cittert–Zernike)")

delta_x = np.linspace(0, 20e-3, 400)
depths = [20e-3, 40e-3, 70e-3]
src_size = 1e-3  # 1 mm incoherent source region (cavitation cloud)
lam = C0 / F0

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#2166ac", "#4dac26", "#d6604d"]
for z, col in zip(depths, colors):
    mu = np.asarray(kw.van_cittert_zernike_coherence(delta_x, src_size, z, lam))
    ax.plot(delta_x * 1e3, np.abs(mu), color=col, label=f"z = {z*1e3:.0f} mm")
    # Mark coherence length (first zero)
    xc = lam * z / src_size
    ax.axvline(xc * 1e3, color=col, lw=0.8, ls="--", alpha=0.6)

ax.set_xlabel("Element separation Δx (mm)")
ax.set_ylabel("|μ(Δx)| — spatial coherence")
ax.set_title("Van Cittert–Zernike coherence\n"
             "Dashed lines: coherence length $\\Delta x_c = \\lambda z / L_s$")
ax.legend()
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.05)
fig.tight_layout()
savefig("fig03_vcz_coherence")
plt.close(fig)


# ── Figure 04: Eigenspace beamformer singular-value structure ─────────────────
print("[fig04] Eigenspace PAM: singular-value decomposition")

N_SRC = 5
# Rust computes Theorem 22.2's deterministic signal/noise eigenvalue split.
# For a Hermitian PSD CSD matrix the eigenvalues equal the singular values.
sv = np.asarray(kw.eigenspace_covariance_eigenvalues(N_EL, N_SRC, 10.0, 1.0))

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(np.arange(1, len(sv) + 1), sv, "o-", color="#2166ac",
            markersize=3, lw=1.2, label="Singular values of R")
ax.axhline(
    sv[N_SRC], color="#d6604d", lw=1.0, ls="--",
    label=f"Signal/noise boundary (rank {N_SRC})",
)
ax.axvspan(
    1, N_SRC + 0.5, alpha=0.08, color="#4dac26",
    label=f"Signal subspace ({N_SRC} sources)",
)
ax.axvspan(N_SRC + 0.5, N_EL, alpha=0.05, color="#d6604d", label="Noise subspace")
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value (log scale)")
ax.set_xlim(1, N_EL)
ax.set_title("Cross-spectral density matrix singular values\n"
             "Eigenspace PAM uses signal subspace to improve spatial resolution")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
savefig("fig04_eigenspace_svd")
plt.close(fig)


# ── Figure 05: Transcranial attenuation effect on PAM SNR ────────────────────
print("[fig05] Transcranial attenuation and phase aberration on PAM SNR")

# Two-way skull insertion loss via kw.skull_insertion_loss_two_way_db
# IL_two(f) = 2 · α₀ · f^1.2 · d   (Fry & Barger 1978; Connor et al. 2002)
f_mhz = np.linspace(0.25, 3.0, 300)
d_cm = SKULL_THICK * 100.0
il_two = np.asarray(kw.skull_insertion_loss_two_way_db(f_mhz, d_cm, SKULL_ALPHA))
il_one = il_two / 2.0

# Phase aberration coherence loss via kw.strehl_ratio (Maréchal approximation):
#   SR = exp(-σ_φ²),   CF_phase_dB = 10·log10(SR)  [dB, ≤ 0]
# σ_φ(f) = σ₀ · f/f₀  (linear phase shift growth with frequency).
sigma_phi0 = 1.0  # radian RMS at 1 MHz (severe skull aberration)
sigma_phi = sigma_phi0 * f_mhz / 1.0
cf_phase_db = np.array([10.0 * np.log10(kw.strehl_ratio(s) + 1e-30)
                         for s in sigma_phi])  # dB, ≤ 0

snr_reduction = il_two + (-cf_phase_db)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
ax1, ax2 = axes

ax1.plot(f_mhz, il_one, color="#2166ac", label="One-way (receive)")
ax1.plot(f_mhz, il_two, color="#d6604d", label="Two-way (Tx + Rx)", ls="--")
ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("Insertion loss (dB)")
ax1.set_title("Skull attenuation\n$\\alpha(f)=\\alpha_0 f^{1.2}$")
ax1.legend()
ax1.set_xlim(0.25, 3.0)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

ax2.plot(f_mhz, il_two, color="#d6604d", label="Attenuation loss", lw=1.3)
ax2.plot(f_mhz, -cf_phase_db, color="#4dac26", label="Phase aberration loss", lw=1.3, ls="--")
ax2.plot(f_mhz, snr_reduction, color="#000", label="Total SNR reduction", lw=1.8)
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("SNR reduction (dB)")
ax2.set_title("Transcranial PAM SNR budget\n"
              f"skull d = {d_cm:.1f} cm, $\\sigma_\\phi(1\\,\\mathrm{{MHz}})$={sigma_phi0:.1f} rad")
ax2.legend()
ax2.set_xlim(0.25, 3.0)

fig.tight_layout()
savefig("fig05_transcranial_snr_budget")
plt.close(fig)


# ── Figure 06: Cavitation dose accumulation — stable vs inertial ──────────────
print("[fig06] Cavitation dose accumulation (stable vs inertial)")

t = np.linspace(0.0, 10.0, 2000)  # 10-second treatment window
dose_trace = kw.passive_cavitation_dose_fixture(t, 1.0, 100e-3, seed=0)
d_sc = np.asarray(dose_trace["stable_dose"])
d_ic1 = np.asarray(dose_trace["inertial_trial1_dose"])
d_ic2 = np.asarray(dose_trace["inertial_trial2_dose"])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, d_sc, color="#2166ac", lw=1.8, label="Stable cavitation dose (deterministic)")
ax.plot(t, d_ic1, color="#d6604d", lw=1.2, label="Inertial cavitation dose (trial 1)")
ax.plot(t, d_ic2, color="#d6604d", lw=1.2, ls="--", alpha=0.6, label="Inertial cavitation dose (trial 2)")
ax.set_xlabel("Treatment time (s)")
ax.set_ylabel("Normalised cumulative dose")
ax.set_title("Cavitation dose accumulation\n"
             "SC: deterministic linear growth; IC: compound Poisson (stochastic)")
ax.legend()
ax.set_xlim(0, 10)
fig.tight_layout()
savefig("fig06_cavitation_dose_accumulation")
plt.close(fig)

print("[ch23] All figures complete.")
