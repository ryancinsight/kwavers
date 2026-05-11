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
Requires: numpy, matplotlib, scipy

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
from scipy.linalg import svd
from scipy.signal import welch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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

RNG = np.random.default_rng(seed=0)


# ── Figure 01: Cavitation emission spectra ────────────────────────────────────
print("[fig01] Cavitation emission spectra (stable vs inertial)")

def _cavitation_spectrum(
    f: np.ndarray, f0: float, kind: str, snr_db: float = 30.0
) -> np.ndarray:
    """
    Analytical power-spectral model for cavitation emission (Coviello 2015 §II-A).

    Stable cavitation (SC): energy concentrated at harmonics of f₀ and the
    sub-harmonic f₀/2; inter-harmonic noise floor is low.

    Inertial cavitation (IC): broadband emission floor elevated by ~20 dB
    relative to SC; harmonic peaks still present but broadened.

    Model:
        S_sc(f) = Σₙ A_n · Lorentz(f; n·f₀, Δf_n)  +  A_sub · Lorentz(f; f₀/2, Δf_sub)
        S_ic(f) = S_sc(f)  +  S_bb(f)
        S_bb(f) = B / (1 + (f/f_c)^4)   (Butterworth broadband envelope)
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    harmonics = np.arange(1, 6)
    amplitudes = 1.0 / harmonics**1.5  # decreasing harmonic envelope
    delta_f = 0.02 * f0  # linewidth ~ 2 % of f0 for stable SC

    def lorentz(f: np.ndarray, fc: float, bw: float) -> np.ndarray:
        return (bw / 2.0) ** 2 / ((f - fc) ** 2 + (bw / 2.0) ** 2)

    spec = np.zeros_like(f)
    for n, A in zip(harmonics, amplitudes):
        spec += A * lorentz(f, n * f0, delta_f)
    # sub-harmonic
    spec += 0.4 * lorentz(f, 0.5 * f0, 0.5 * delta_f)

    if kind == "ic":
        f_c = 3.0 * f0
        broadband = 0.15 / (1.0 + (f / f_c) ** 4)
        spec += broadband

    noise_floor = np.max(spec) / snr_lin
    spec += noise_floor
    return 10.0 * np.log10(spec / np.max(spec))

f_vec = np.linspace(0.1e6, 5.5e6, 4000)
sc_db = _cavitation_spectrum(f_vec, F0, "sc")
ic_db = _cavitation_spectrum(f_vec, F0, "ic")

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


# ── Figure 02: DAS sensitivity map vs focal depth ─────────────────────────────
print("[fig02] Passive DAS sensitivity map vs focal depth")

def _das_sensitivity(x_src: float, z_src: float,
                     el_positions: np.ndarray, c: float, fs: float) -> float:
    """
    Coherent DAS gain for a point source at (x_src, z_src).

    Each element delay τ_i = r_i / c where r_i is the element-to-source distance.
    After delay-and-sum, the coherent pressure amplitude is Σ_i w_i · p(t - τ_i).
    For a unit-amplitude CW source the coherent output amplitude equals
    |Σ_i w_i · exp(j ω τ_i)| which, for uniform weights, is N only at the
    steering focus and decays as a sinc-like spatial pattern away from it.

    Here we compute the Fraunhofer diffraction integral numerically on a 2-D
    lateral × depth grid for a linear aperture.
    """
    r = np.sqrt((el_positions - x_src) ** 2 + z_src ** 2)
    phase = 2.0 * np.pi * F0 * r / c
    return float(np.abs(np.sum(np.exp(-1j * phase))) / len(el_positions))

el_x = (np.arange(N_EL) - (N_EL - 1) / 2.0) * PITCH
x_range = np.linspace(-15e-3, 15e-3, 120)
z_range = np.linspace(5e-3, 80e-3, 140)
sens = np.zeros((len(z_range), len(x_range)))
for iz, z in enumerate(z_range):
    for ix, x in enumerate(x_range):
        sens[iz, ix] = _das_sensitivity(x, z, el_x, C0, FS)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(
    20 * np.log10(sens / sens.max() + 1e-12),
    extent=[x_range[0] * 1e3, x_range[-1] * 1e3, z_range[-1] * 1e3, z_range[0] * 1e3],
    aspect="auto", cmap="hot", vmin=-40, vmax=0,
)
cb = fig.colorbar(im, ax=ax)
cb.set_label("Sensitivity (dB re peak)")
ax.set_xlabel("Lateral position (mm)")
ax.set_ylabel("Depth (mm)")
ax.set_title(f"Passive DAS sensitivity — {N_EL}-element aperture, pitch {PITCH*1e3:.1f} mm\n"
             f"$f_0$ = {F0/1e6:.1f} MHz,  $c$ = {C0:.0f} m/s")
fig.tight_layout()
savefig("fig02_das_sensitivity_map")
plt.close(fig)


# ── Figure 03: Spatial coherence and van Cittert–Zernike theorem ──────────────
print("[fig03] Spatial coherence function (van Cittert–Zernike)")

def _vcz_coherence(delta_x: np.ndarray, z: float, L_src: float) -> np.ndarray:
    """
    Van Cittert–Zernike (VCZ) theorem for an incoherent planar source of
    lateral extent L_src at depth z (far-field / paraxial regime):

        μ(Δx) = sinc(L_src Δx / (λ z))     where sinc(u) = sin(πu)/(πu)

    Physical interpretation: the coherence length Δx_c = λ z / L_src sets the
    minimum aperture element spacing for independent measurements.
    Reference: Goodman (2015) Speckle Phenomena §5.2.
    """
    lam = C0 / F0
    u = L_src * delta_x / (lam * z)
    return np.sinc(u)  # numpy sinc is normalised: sinc(u) = sin(πu)/(πu)

delta_x = np.linspace(0, 20e-3, 400)
depths = [20e-3, 40e-3, 70e-3]
src_size = 1e-3  # 1 mm incoherent source region (cavitation cloud)

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#2166ac", "#4dac26", "#d6604d"]
for z, col in zip(depths, colors):
    mu = _vcz_coherence(delta_x, z, src_size)
    ax.plot(delta_x * 1e3, np.abs(mu), color=col, label=f"z = {z*1e3:.0f} mm")
    # Mark coherence length (first zero)
    lam = C0 / F0
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

def _build_csd_matrix(
    n_el: int, n_src: int, n_snap: int, signal_power: float, noise_power: float
) -> np.ndarray:
    """
    Simulate a cross-spectral density (CSD) matrix for PAM (Arnal 2017 §III).

    Model: K incoherent point sources, each with a steering vector a_k ∈ ℂ^N.
    CSD matrix:  R = Σ_k σ_k² a_k a_k^H  +  σ_n² I
    Noise is circular complex Gaussian, spatially white.

    Computed as sample CSD from n_snap simulated snapshots.
    """
    el_pos = (np.arange(n_el) - (n_el - 1) / 2.0) * PITCH
    src_x = RNG.uniform(-5e-3, 5e-3, n_src)
    src_z = RNG.uniform(40e-3, 60e-3, n_src)

    # Steering vectors
    A = np.zeros((n_el, n_src), dtype=complex)
    for k in range(n_src):
        r = np.sqrt((el_pos - src_x[k]) ** 2 + src_z[k] ** 2)
        A[:, k] = np.exp(1j * 2.0 * np.pi * F0 * r / C0) / np.sqrt(n_el)

    # Data matrix X = A s + n  (n_snap snapshots)
    s = RNG.standard_normal((n_src, n_snap)) + 1j * RNG.standard_normal((n_src, n_snap))
    s *= np.sqrt(signal_power / 2.0)
    noise = RNG.standard_normal((n_el, n_snap)) + 1j * RNG.standard_normal((n_el, n_snap))
    noise *= np.sqrt(noise_power / 2.0)
    X = A @ s + noise

    R = (X @ X.conj().T) / n_snap
    return R

R = _build_csd_matrix(n_el=N_EL, n_src=5, n_snap=256, signal_power=10.0, noise_power=1.0)
_, sv, _ = svd(R)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(np.arange(1, len(sv) + 1), sv, "o-", color="#2166ac",
            markersize=3, lw=1.2, label="Singular values of R")
ax.axhline(sv[5], color="#d6604d", lw=1.0, ls="--", label="Signal/noise boundary (rank 5)")
ax.axvspan(1, 5.5, alpha=0.08, color="#4dac26", label="Signal subspace (5 sources)")
ax.axvspan(5.5, N_EL, alpha=0.05, color="#d6604d", label="Noise subspace")
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

def _skull_transmission_loss(f_MHz: np.ndarray, d_cm: float) -> np.ndarray:
    """
    One-way skull insertion loss [dB]:
        IL(f) = α₀ f^1.2 · d
    Two-way (transmit + receive) = 2 × IL.
    Coefficients: Fry & Barger (1978); Connor et al. (2002).
    """
    return SKULL_ALPHA * (f_MHz ** 1.2) * d_cm

f_mhz = np.linspace(0.25, 3.0, 300)
d_cm = SKULL_THICK * 100.0
il_one = _skull_transmission_loss(f_mhz, d_cm)
il_two = 2.0 * il_one

# PAM SNR reduction = two-way skull loss + phase-aberration coherence loss.
# Phase aberration modelled as Gaussian random phase σ_φ ~ N(0, σ²) per element.
# Coherence factor due to phase aberration (Maréchal approximation):
#   CF_phase = exp(-σ_φ²)
# where σ_φ scales with f:  σ_φ(f) = σ₀ · f/f₀ (linear phase shift growth).
sigma_phi0 = 1.0  # radian RMS at 1 MHz (severe skull aberration)
sigma_phi = sigma_phi0 * f_mhz / 1.0
cf_phase_db = -10.0 * np.log10(np.exp(1.0)) * sigma_phi ** 2  # dB, ≤0

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

def _cavitation_dose(
    t_s: np.ndarray, prf: float, tau_p: float, kind: str
) -> np.ndarray:
    """
    Cumulative cavitation dose models (Gyöngy & Coussios 2010 §III):

    Stable cavitation (SC) dose  D_sc:
        Increments at each pulse by the inertial cavitation dose-equivalent
        computed from sub-harmonic emission amplitude.
        Here modelled as a linearly accumulating quantity (coherent emissions
        reinforce; no catastrophic collapse events).
        D_sc(t) ∝  Σ_{pulse k} A_sc(k)  ≈ constant rate (steady-state cavitation).

    Inertial cavitation (IC) dose  D_ic:
        Stochastic events (individual bubble collapses); amplitude distribution
        follows a heavy tail.  Modelled as a Poisson process with rate λ_ic;
        each event contributes a random collapse energy drawn from an
        exponential distribution.
        D_ic(t) is a compound Poisson process; mean grows linearly but
        fluctuations are large.
    """
    pulse_times = np.arange(0.0, t_s[-1], 1.0 / prf)
    dose = np.zeros_like(t_s)

    if kind == "sc":
        # Deterministic linear accumulation (sustained coherent oscillation)
        rate = 1.0  # normalised dose per pulse
        for tp in pulse_times:
            dose[t_s >= tp] += rate * tau_p * prf
        dose /= dose[-1] if dose[-1] > 0 else 1.0
    else:
        # Compound Poisson: stochastic collapse events
        lambda_ic = 0.3 * prf  # mean IC events per second
        for tp in pulse_times:
            n_events = RNG.poisson(lambda_ic / prf)
            if n_events > 0:
                energies = RNG.exponential(scale=1.0, size=n_events)
                mask = t_s >= tp
                dose[mask] += energies.sum()
        if dose[-1] > 0:
            dose /= dose[-1]

    return dose

t = np.linspace(0.0, 10.0, 2000)  # 10-second treatment window
d_sc = _cavitation_dose(t, prf=1.0, tau_p=100e-3, kind="sc")
# Run IC twice to show stochastic spread
d_ic1 = _cavitation_dose(t, prf=1.0, tau_p=100e-3, kind="ic")
d_ic2 = _cavitation_dose(t, prf=1.0, tau_p=100e-3, kind="ic")

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
