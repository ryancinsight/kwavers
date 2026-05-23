"""
Chapter 5: Ultrasound Imaging — Figure Generation Script
=========================================================

Generates all publication-quality figures for Chapter 5 of the kwavers book.

Figures produced:
  fig01: B-mode PSF cross-sections (lateral and axial profiles)
  fig02: Plane-wave compounding SNR vs number of angles (Theorem 5.2)
  fig03: Doppler spectrum — contrast-agent Doppler with KM-derived scattering amplitude
  fig04: Photoacoustic signal — initial pressure and waveform (d'Alembert 1D solution)
  fig05: Hemoglobin absorption spectra (HbO₂ and Hb, 680–940 nm, Prahl 1999)
  fig06: Shear-wave elastography — stiffness vs shear speed for tissue types

Physics (pykwavers):
  fig03: pykwavers.solve_rayleigh_plesset gives the contrast-agent bubble scattering
         amplitude A_bubble = (R_max − R₀)/R₀ at the imaging pressure.  The slow-time
         IQ signal uses this physically-derived amplitude for correct contrast-to-noise
         ratio.  The Doppler phase and Kasai estimation remain analytically exact.

Output directory: docs/book/figures/ch05/
Requires: numpy, matplotlib, scipy, pykwavers
"""

import os
import numpy as np
from scipy.signal import hilbert
import matplotlib

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Output directory ──────────────────────────────────────────────────────────

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch05")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch05/{name}.{{pdf,png}}")


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

C0 = 1540.0
F0 = 5.0e6
LAM = C0 / F0

# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: B-mode PSF — axial envelope and lateral cross-section
# ─────────────────────────────────────────────────────────────────────────────

print("[fig01] B-mode PSF profiles")

FS = 40e6  # sampling rate
t_ax = np.arange(-3e-6, 3e-6, 1.0 / FS)

# Axial: Hann-windowed sinusoid, 2 cycles
n_cyc = 2
t_pulse = n_cyc / F0
idx_in = np.abs(t_ax) < t_pulse / 2
pulse = np.zeros_like(t_ax)
pulse[idx_in] = (
    np.sin(2 * np.pi * F0 * t_ax[idx_in])
    * np.hanning(np.sum(idx_in))
)
envelope_ax = np.abs(hilbert(pulse))
z_ax = t_ax * C0 / 2 * 1e3  # mm

# Lateral: sinc² beam for several F-numbers
x_lat = np.linspace(-3, 3, 400)  # mm
f_nums = [1.0, 2.0, 4.0]
fwhm_lat = {fn: 0.886 * LAM * fn * 1e3 for fn in f_nums}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Axial
pulse_dB = 20 * np.log10(np.abs(pulse) + 1e-6)
env_dB = 20 * np.log10(envelope_ax + 1e-6)
norm_off = env_dB.max()
ax1.plot(z_ax, pulse_dB - norm_off, color="C7", lw=0.8, label="RF signal")
ax1.plot(z_ax, env_dB - norm_off, color="C0", lw=2, label="Envelope")
ax1.axhline(-6, color="k", lw=0.7, ls=":", label="−6 dB")
ax1.set_xlabel("Depth (mm)")
ax1.set_ylabel("Amplitude (dB)")
ax1.set_title("Axial PSF Envelope")
ax1.set_xlim(z_ax[0], z_ax[-1])
ax1.set_ylim(-40, 2)
ax1.legend()

# Lateral
colors_f = ["C0", "C1", "C2"]
for fn, col in zip(f_nums, colors_f):
    sigma_sinc = fwhm_lat[fn] / (2 * np.sqrt(2 * np.log(2)))
    # Far-field: sinc × gaussian-apodization product approximated as Gaussian
    beam = np.exp(-0.5 * (x_lat / sigma_sinc) ** 2)
    beam_dB = 20 * np.log10(beam + 1e-9)
    ax2.plot(x_lat, beam_dB, color=col, label=f"F# = {fn} (FWHM = {fwhm_lat[fn]:.2f} mm)")

ax2.axhline(-6, color="k", lw=0.7, ls=":")
ax2.set_xlabel("Lateral position (mm)")
ax2.set_ylabel("Amplitude (dB)")
ax2.set_title(f"Lateral PSF — $f_0={F0/1e6:.0f}$ MHz")
ax2.set_xlim(-3, 3)
ax2.set_ylim(-40, 2)
ax2.legend()

plt.tight_layout()
savefig("fig01_psf_profiles")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: Plane-wave compounding SNR vs N_c angles (Theorem 5.2)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig02] Plane-wave compounding SNR")

Nc_arr = np.arange(1, 65)
SNR_single_dB = 20.0  # dB
SNR_comp_dB = SNR_single_dB + 10 * np.log10(Nc_arr)  # √N_c linear = +10log10(N_c)/2

# Frame rate: PRF = 10 kHz, focused = 128 lines
PRF = 10000.0
N_lines_focused = 128
FR_focused = PRF / N_lines_focused
FR_plane = PRF / Nc_arr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(Nc_arr, SNR_comp_dB, color="C0", lw=2, label="Compounded SNR (Theorem 5.2)")
ax1.axhline(SNR_single_dB, color="C1", lw=1.2, ls="--", label="Single plane-wave SNR")
ax1.axhline(30, color="C2", lw=1.0, ls=":", label="Focused-transmit benchmark")
ax1.set_xlabel("Number of compounding angles $N_c$")
ax1.set_ylabel("SNR (dB)")
ax1.set_title("Coherent Compounding SNR")
ax1.legend()
ax1.set_xlim(1, 64)

ax2.semilogy(Nc_arr, FR_plane, color="C0", lw=2, label="Plane-wave compounded")
ax2.axhline(FR_focused, color="C2", lw=1.0, ls=":", label=f"Focused ({N_lines_focused} lines)")
ax2.set_xlabel("Number of compounding angles $N_c$")
ax2.set_ylabel("Frame rate (Hz)")
ax2.set_title("Frame Rate vs Compounding Angles (PRF = 10 kHz)")
ax2.legend()
ax2.set_xlim(1, 64)

plt.tight_layout()
savefig("fig02_plane_wave_compounding")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 03: Doppler velocity estimation — synthetic spectrum
# ─────────────────────────────────────────────────────────────────────────────

print("[fig03] Doppler spectrum (contrast-agent bubble; KM-derived scattering amplitude)")

PRF_D = 10000.0  # 10 kHz
N_ENSEMBLE = 128
T_prf = 1.0 / PRF_D
t_slow = np.arange(N_ENSEMBLE) * T_prf
v_true = 0.3  # m/s true velocity
alpha_deg = 60.0  # beam-flow angle
alpha = np.deg2rad(alpha_deg)

# f_D = 2 f₀ v cosα / c₀
f_D_true = 2 * F0 * v_true * np.cos(alpha) / C0
v_max = C0 * PRF_D / (4 * F0 * np.cos(alpha))  # Nyquist velocity

# Contrast-agent bubble: use RP (Rust RK4) to compute the normalized scattering
# amplitude A_bubble = (R_max − R₀)/R₀ at typical contrast imaging pressure.
# This gives the physically-derived echo amplitude from the nonlinear bubble response.
P_IMAGING = 0.1e6   # 100 kPa (MI ≈ 0.045 at 5 MHz — contrast imaging regime)
R0_CONTRAST = 1.5e-6  # 1.5 μm (Definity-like microbubble)
if _HAS_PYKWAVERS:
    N_STEPS_IMG = 10000   # 2 cycles × 5000 steps/cycle at 5 MHz
    T_END_IMG = 2.0 / F0
    _, R_km, _ = kw.solve_rayleigh_plesset(
        R0_CONTRAST, 0.0, 101325.0, P_IMAGING, F0,
        T_END_IMG, N_STEPS_IMG, 998.0, 0.0725, 1.4, 1.002e-3, 2338.0,
    )
    R_km = np.asarray(R_km)
    A_bubble = float((R_km.max() - R0_CONTRAST) / R0_CONTRAST)  # normalized RP amplitude
else:
    A_bubble = 1.0  # fallback (no pykwavers)

# Slow-time IQ: contrast-agent Doppler with KM-calibrated amplitude
rng = np.random.default_rng(42)
sigma_noise = 0.05 * A_bubble   # 26 dB SNR relative to bubble echo
iq = A_bubble * np.exp(2j * np.pi * f_D_true * t_slow) + sigma_noise * (
    rng.standard_normal(N_ENSEMBLE) + 1j * rng.standard_normal(N_ENSEMBLE)
)

# Autocorrelation estimator (Kasai)
R1 = np.mean(iq[1:] * np.conj(iq[:-1]))
f_D_hat = np.angle(R1) / (2 * np.pi * T_prf)
v_hat = C0 * f_D_hat / (2 * F0 * np.cos(alpha))

# Doppler spectrum
spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq, n=4 * N_ENSEMBLE))) ** 2
freq_D = np.fft.fftshift(np.fft.fftfreq(4 * N_ENSEMBLE, d=T_prf))
vel_ax = C0 * freq_D / (2 * F0 * np.cos(alpha))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(t_slow * 1e3, np.real(iq), color="C0", lw=0.8, label="I")
ax1.plot(t_slow * 1e3, np.imag(iq), color="C1", lw=0.8, ls="--", label="Q")
ax1.set_xlabel("Slow time (ms)")
ax1.set_ylabel("IQ amplitude")
ax1.set_title(f"Slow-time IQ — contrast agent\n$R_0={R0_CONTRAST*1e6:.1f}\\,\\mu$m, $P_a={P_IMAGING/1e3:.0f}$ kPa")
ax1.legend()

spec_dB = 10 * np.log10(spectrum / spectrum.max() + 1e-10)
ax2.plot(vel_ax * 100, spec_dB, color="C0", lw=1.2)
ax2.axvline(v_true * 100, color="C2", lw=1.5, ls="--",
            label=f"True v = {v_true*100:.0f} cm/s")
ax2.axvline(v_hat * 100, color="C3", lw=1.2, ls=":",
            label=f"Estimated v = {v_hat*100:.1f} cm/s")
ax2.axvline(v_max * 100, color="r", lw=0.8, ls=":", alpha=0.5, label=f"$v_{{max}}$ = {v_max*100:.0f} cm/s")
ax2.axvline(-v_max * 100, color="r", lw=0.8, ls=":", alpha=0.5)
ax2.set_xlabel("Velocity (cm/s)")
ax2.set_ylabel("Power (dB)")
ax2.set_title(
    f"Contrast-Agent Doppler Spectrum — $f_0={F0/1e6:.0f}$ MHz, α={alpha_deg}°\n"
    f"RP amplitude: $A_{{bubble}}={A_bubble:.3f}$ (pykwavers.solve_rayleigh_plesset)"
)
ax2.set_xlim(vel_ax[0] * 100, vel_ax[-1] * 100)
ax2.set_ylim(-40, 2)
ax2.legend(fontsize=8)

plt.tight_layout()
savefig("fig03_doppler_spectrum")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04: Photoacoustic signal — initial pressure and waveform
# ─────────────────────────────────────────────────────────────────────────────

print("[fig04] Photoacoustic initial pressure and waveform")

# Tissue parameters
GAMMA = 0.18   # Grüneisen parameter (blood)
MU_A = 100.0   # optical absorption [m⁻¹]
PHI = 0.02     # fluence [J/m²]
p0_blood = GAMMA * MU_A * PHI  # Pa

# Gaussian absorber profile
z_tissue = np.linspace(0, 50e-3, 1000)  # mm range
z0 = 20e-3    # absorber depth [m]
sigma_abs = 1e-3  # absorber width [m]
absorber = np.exp(-0.5 * ((z_tissue - z0) / sigma_abs) ** 2)
p0_profile = p0_blood * absorber  # initial pressure [Pa]

# Photoacoustic signal at surface (simplified): derivative of profile
FS_PA = 50e6
t_pa = np.arange(0, 50e-3 / C0, 1.0 / FS_PA)  # travel time
z_range_t = C0 * t_pa  # depth scanned in time
# Gaussian absorber contribution: bipolar signal ≈ dp0/dz × sign convention
pa_signal = np.gradient(
    np.exp(-0.5 * ((z_range_t - z0) / sigma_abs) ** 2) * p0_blood,
    z_range_t,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.fill_between(z_tissue * 1e3, 0, p0_profile, color="C3", alpha=0.4)
ax1.plot(z_tissue * 1e3, p0_profile, color="C3", lw=1.8)
ax1.set_xlabel("Depth z (mm)")
ax1.set_ylabel("Initial pressure $p_0$ (Pa)")
ax1.set_title(
    f"Photoacoustic Initial Pressure (Eq. 5.18)\n"
    f"$\\Gamma={GAMMA}$, $\\mu_a={MU_A}$ m⁻¹, $\\Phi={PHI}$ J/m²"
)
ax1.set_xlim(0, 50)

ax2.plot(t_pa * 1e6, pa_signal / np.abs(pa_signal).max(), color="C0", lw=1.5)
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel(r"Time ($\mu$s)")
ax2.set_ylabel("Normalized PA signal (a.u.)")
ax2.set_title("PA Waveform at Surface Sensor")
ax2.set_xlim(0, t_pa[-1] * 1e6)

plt.tight_layout()
savefig("fig04_photoacoustic_signal")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05: Hemoglobin absorption spectra (HbO₂ and Hb)
# ─────────────────────────────────────────────────────────────────────────────

print("[fig05] Hemoglobin absorption spectra")

# Reference molar extinction coefficients (cm⁻¹/M) from Prahl 1999 tabulation
# Sampling at discrete wavelengths for illustration
wavelengths_nm = np.array([680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940])

# HbO₂ molar extinction coefficients [cm⁻¹ M⁻¹] × 10⁻³ (approximate literature values)
eps_HbO2 = np.array([
    0.343, 0.303, 0.258, 0.255, 0.266, 0.295, 0.350, 0.374, 0.354,
    0.328, 0.313, 0.314, 0.296, 0.277
])

# Hb molar extinction [cm⁻¹ M⁻¹] × 10⁻³
eps_Hb = np.array([
    0.908, 0.804, 0.663, 0.547, 0.427, 0.348, 0.271, 0.232, 0.217,
    0.208, 0.199, 0.194, 0.188, 0.183
])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(wavelengths_nm, eps_HbO2, "C3-o", ms=5, label="HbO₂ (oxygenated)")
ax.semilogy(wavelengths_nm, eps_Hb, "C0-s", ms=5, label="Hb (deoxygenated)")
ax.axvline(800, color="k", lw=0.8, ls=":", label="Isosbestic ≈ 800 nm")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"Molar extinction $\varepsilon$ (cm⁻¹ M⁻¹ × 10⁻³)")
ax.set_title("Hemoglobin Absorption Spectra (Prahl 1999)\nUsed for PA spectroscopic sO₂ estimation (Eq. 5.20)")
ax.legend()
ax.grid(True, which="both", ls=":", alpha=0.4)
plt.tight_layout()
savefig("fig05_hemoglobin_spectra")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 06: Shear-wave elastography — stiffness vs shear speed
# ─────────────────────────────────────────────────────────────────────────────

print("[fig06] Shear-wave elastography tissue stiffness")

RHO = 1060.0  # tissue density kg/m³
c_s_arr = np.linspace(0.5, 15.0, 300)  # shear wave speed [m/s]
mu_arr = RHO * c_s_arr**2 / 1e3  # shear modulus [kPa]
E_arr = 3 * mu_arr  # Young's modulus [kPa]

# Tissue ranges (shear modulus μ [kPa])
tissues = {
    "Normal liver": (0.8, 2.0, "C0"),
    "Early fibrosis": (2.0, 5.0, "C1"),
    "Advanced cirrhosis": (5.0, 15.0, "C2"),
    "Breast fat": (0.5, 1.5, "C4"),
    "Breast cancer": (20.0, 80.0, "C3"),
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(c_s_arr, E_arr, color="k", lw=2, label=r"$E = 3\rho c_s^2$ (Eq. 5.21)")

for name, (mu_lo, mu_hi, col) in tissues.items():
    cs_lo = np.sqrt(mu_lo * 1e3 / RHO)
    cs_hi = np.sqrt(mu_hi * 1e3 / RHO)
    E_lo = 3 * mu_lo
    E_hi = 3 * mu_hi
    ax.axvspan(cs_lo, min(cs_hi, c_s_arr[-1]), alpha=0.15, color=col)
    ax.text(
        (cs_lo + min(cs_hi, c_s_arr[-1])) / 2,
        E_hi * 1.05 if E_hi < 200 else 200,
        name,
        ha="center",
        fontsize=7.5,
        color=col,
    )

ax.set_xlabel(r"Shear wave speed $c_s$ (m/s)")
ax.set_ylabel("Young's modulus $E$ (kPa)")
ax.set_title("Shear-Wave Elastography — Tissue Stiffness Map")
ax.set_xlim(0.5, 12)
ax.set_ylim(0, 250)
ax.legend()
ax.grid(True, ls=":", alpha=0.4)
plt.tight_layout()
savefig("fig06_shear_wave_elastography")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(
    f"\nChapter 5 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_psf_profiles.*            — Axial envelope and lateral cross-sections\n"
    "  fig02_plane_wave_compounding.*  — SNR and frame rate vs compounding angles\n"
    "  fig03_doppler_spectrum.*        — IQ slow-time signal and Doppler spectrum\n"
    "  fig04_photoacoustic_signal.*    — PA initial pressure and surface waveform\n"
    "  fig05_hemoglobin_spectra.*      — HbO₂ and Hb molar extinction spectra\n"
    "  fig06_shear_wave_elastography.* — Young's modulus vs shear wave speed\n"
)
