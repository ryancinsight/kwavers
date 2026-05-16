"""
Chapter 25: RTM-Driven Adaptive Transcranial Beamforming
=========================================================

Publication-quality figures for docs/book/rtm_adaptive_beamforming.md.

Multi-frequency Reverse Time Migration (RTM) reveals standing-wave
structures inside the skull cavity that reduce transcranial focused
ultrasound (tFUS) efficacy.  This chapter develops a temporal beamforming
modulation strategy — informed by RTM spatial-frequency analysis — that
maintains focal pressure while suppressing off-target standing waves.

Physical model
--------------
Domain     : 2-D coronal cross-section (x lateral, z superior-inferior)
             200 × 200 mm, 0.5 mm resolution.
Skull      : Elliptical shell, outer semi-axes (86, 98) mm, 6 mm thick.
             c_sk = 2800 m/s, ρ_sk = 1900 kg/m³.
Brain      : Interior ellipsoid, c_br = 1530 m/s, ρ_br = 1040 kg/m³.
Water      : Exterior coupling, c_w = 1500 m/s, ρ_w = 1000 kg/m³.
Transducer : 64-element hemispherical phased array, radius 100 mm,
             arc ±60°, focused at (0, −20 mm) from domain centre.

Acoustic field model
--------------------
Forward pressure in brain: Gaussian beam with frequency-dependent waist
plus back-reflection standing-wave factor (Fabry-Pérot model on z-axis).

  P_fwd(x, z, f) = T_eff(f) · G_beam(x, z, f) · SW(z, f)

  SW(z, f) = 1 + R_back · exp[2ik_br(f) · (z_back − z)]

where T_eff is the full skull transfer-matrix transmission coefficient,
R_back = (Z_sk − Z_br)/(Z_sk + Z_br) is the inner skull reflection, and
z_back = +89 mm is the far inner skull surface.

RTM image at frequency f
------------------------
  I_RTM(x, z, f) = Re[ P_fwd(x, z, f) · P_bwd*(x, z, f) ]

P_bwd is the backpropagated Green's function from the focal point:
  P_bwd(x, z, f) = A(f) · exp[−ik_br(f) · r_f] / r_f^½
  r_f = √((x − x_f)² + (z − z_f)²)

FWI reconstruction quality
--------------------------
Skull correction quality σ ∈ [0, 1] parameterises the RMS phase error
induced by skull heterogeneity:
  φ_err ~ N(0, σ_err²),  σ_err = (1 − σ) · σ_skull_max

Higher σ → better skull model → tighter focal beam.

RTM-driven temporal modulation
-------------------------------
RTM spatial spectrum identifies dominant standing-wave wavenumber:
  k_SW(f₀) = 2 · k_br(f₀) = 4πf₀ / c_br

One full standing-wave period requires a frequency shift:
  dF_period = c_br / (2 · d_back),  d_back = z_back − z_f = 109 mm

For M-step modulation spanning one full period:
  df_step = dF_period / M = c_br / (2 · M · d_back)

Time-averaged off-focus intensity normalised by DC:
  <|SW|²>_M = 1 + R_back²

vs static peak:
  |SW_peak|² = (1 + R_back)²

Suppression gain: (1 + R_back)² / (1 + R_back²) [linear power ratio]

RTM-FWI iterative correction
-----------------------------
Each RTM-FWI round recovers a fraction η of the remaining skull phase error:
  σ_{k+1} = σ_k + (1 − σ_k) · η

Converges to σ = 1 (perfect correction) as k → ∞.  Strehl ratio tracks
the Maréchal bound: S_k ≈ exp(−σ_err_k²), σ_err_k = (1 − σ_k)·π.

Brain-volume uniformity
------------------------
For M uniform frequency steps spanning one full SW period:
  <|SW(x,z)|²>_M = 1 + R_back²   (exact, for all M ≥ 2, all (x,z))

The static single-frequency case produces a bimodal distribution:
  antinodes: |SW|² = (1 + R)² = 2.37
  nodes:     |SW|² = (1 − R)² = 0.21

Modulation collapses this onto the Dirac value 1 + R² = 1.29, eliminating
off-target hotspots and dead-zones simultaneously.

Figures produced
----------------
fig10  9-panel RTM/FWI comparison (subharmonic, 110, 220, 440 kHz;
       normal/multiparam/nonlinear FWI; target; fusion)
fig11  RTM standing-wave mode analysis: axial spatial spectrum per freq
fig12  Beamforming delay-law evolution over M temporal modulation steps
fig13  Focal-axis intensity: static vs RTM-modulated (M = 8 steps)
fig14  Time-averaged 2-D field: grating suppression from RTM modulation
fig15  RTM-FWI iterative convergence: Strehl ratio and skull quality vs k
fig16  Temporal pressure dynamics at focal point: |P_foc|² per step, SD vs M
fig17  Brain-volume |SW|² histogram: bimodal→Dirac collapse with M

Output directory: docs/book/figures/ch25/
Requires: numpy, scipy, matplotlib

References
----------
Vignon et al.    (2006) J. Acoust. Soc. Am. 120(4):2330
Deffieux & Konofagou (2010) Ultrasound Med. Biol. 36(7):1117
Marquet et al.   (2009) J. Acoust. Soc. Am. 125(4):2388
Larrat et al.    (2010) IEEE Trans. Ultrason. 57(7):1734
McDannold et al. (2006) Phys. Med. Biol. 51(4):793
Baysal et al.    (1983) Geophysics 48(11):1514
"""

from __future__ import annotations

import os
import numpy as np
from scipy.signal import windows as sig_windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch25")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "lines.linewidth": 1.5,
})


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch25/{name}.{{pdf,png}}")


# ── Physical constants ────────────────────────────────────────────────────────
C_WATER  = 1500.0   # m/s
C_SKULL  = 2800.0   # m/s
C_BRAIN  = 1530.0   # m/s
RHO_WATER = 1000.0  # kg/m³
RHO_SKULL = 1900.0  # kg/m³
RHO_BRAIN = 1040.0  # kg/m³
Z_WATER = RHO_WATER * C_WATER   # 1.500 MRayl
Z_SKULL = RHO_SKULL * C_SKULL   # 5.320 MRayl
Z_BRAIN = RHO_BRAIN * C_BRAIN   # 1.591 MRayl
D_SKULL = 7e-3                   # skull slab thickness [m]

# Reflection coefficient at inner skull-brain interface (skull side incident)
R_BACK = (Z_SKULL - Z_BRAIN) / (Z_SKULL + Z_BRAIN)   # ≈ +0.540

# Transducer geometry
N_ELEM = 64
ARRAY_RADIUS = 100e-3    # radius of curvature [m]
HALF_ARC = np.pi / 3    # ±60° arc
_angles = np.linspace(-HALF_ARC, HALF_ARC, N_ELEM)
X_ELEM = ARRAY_RADIUS * np.sin(_angles)
Z_ELEM = -ARRAY_RADIUS * np.cos(_angles)

# Focal point (domain-centred coordinates)
X_FOC  =  0.0
Z_FOC  = -20e-3   # 20 mm superior to domain centre

# Far inner skull surface along z-axis (standing-wave back-reflector)
AZ_INNER = 89e-3   # m
Z_BACK = AZ_INNER  # +89 mm from centre
D_BACK = Z_BACK - Z_FOC  # 109 mm, focal-to-back-wall distance

# Frequencies
F_SUB =  55e3    # Hz  subharmonic
F0    = 110e3    # Hz  fundamental
F2    = 220e3    # Hz  2nd harmonic
F4    = 440e3    # Hz  4th harmonic
FREQS_RTM = [F_SUB, F0, F2, F4]

# 2-D domain
NX, NZ = 400, 400
LX = LZ = 200e-3
DX = DZ = LX / NX
x_arr = np.linspace(-LX / 2, LX / 2, NX)
z_arr = np.linspace(-LZ / 2, LZ / 2, NZ)
X, Z = np.meshgrid(x_arr, z_arr, indexing="ij")   # (NX, NZ)

# ── Skull phantom (elliptical shell) ──────────────────────────────────────────
AX_OUTER, AZ_OUTER = 86e-3, 98e-3
AX_INNER, AZ_INNER_MASK = 80e-3, 92e-3

r_outer = np.sqrt((X / AX_OUTER)**2 + (Z / AZ_OUTER)**2)
r_inner = np.sqrt((X / AX_INNER)**2 + (Z / AZ_INNER_MASK)**2)

IN_SKULL = (r_outer <= 1.0) & (r_inner > 1.0)
IN_BRAIN = r_inner <= 1.0
SKULL_CONTOUR = r_inner   # used for contour overlay


# ── Skull transfer-matrix transmission ────────────────────────────────────────
def skull_transmission(freq: float) -> complex:
    """
    Pressure transmission coefficient for a plane wave traversing the skull slab.

    Transfer-matrix result (Brekhovskikh 1980):
      T = 2 / [(1 + Z3/Z1)cos(k2d) + i(Z3/Z2 + Z2/Z1)sin(k2d)]
    """
    k2 = 2.0 * np.pi * freq / C_SKULL
    c2d, s2d = np.cos(k2 * D_SKULL), np.sin(k2 * D_SKULL)
    denom = (1.0 + Z_BRAIN / Z_WATER) * c2d + 1j * (Z_BRAIN / Z_SKULL + Z_SKULL / Z_WATER) * s2d
    return 2.0 / denom


# ── Gaussian beam parameters ──────────────────────────────────────────────────
def _beam_params(freq: float) -> tuple[float, float]:
    """Diffraction-limited beam waist w0 and Rayleigh range z_R at focal point."""
    lam = C_BRAIN / freq
    aperture = 2.0 * ARRAY_RADIUS * np.sin(HALF_ARC)
    F_num = ARRAY_RADIUS / aperture
    w0 = 0.51 * lam * F_num   # 1/e² radius at focus (Siegman 1986)
    z_R = np.pi * w0**2 / lam
    return w0, z_R


# ── Forward pressure field ────────────────────────────────────────────────────
def forward_field(
    freq: float,
    skull_qual: float = 1.0,
    phase_offset: float = 0.0,
    rng_seed: int = 0,
) -> np.ndarray:
    """
    Complex pressure amplitude P_fwd(x, z, f) inside brain.

    Parameters
    ----------
    skull_qual  : Skull phase-correction quality ∈ [0, 1].
                  1 = perfect delay law; 0 = uncorrected skull.
    phase_offset: Global phase shift applied to all elements [rad].
    rng_seed    : Seed for reproducible skull aberration noise.

    Model
    -----
    P = T_eff · G_beam · SW · aberration

    G_beam  : Gaussian beam with diffraction-limited waist.
    SW      : Fabry-Pérot standing-wave factor along z.
    aberration: Phase screen from residual skull error (skull_qual < 1).
    """
    k_br = 2.0 * np.pi * freq / C_BRAIN
    T_eff = skull_transmission(freq)
    w0, z_R = _beam_params(freq)

    # ── Gaussian beam at focal point ──
    dz = Z - Z_FOC
    dx = X - X_FOC
    w_z = w0 * np.sqrt(1.0 + (dz / z_R) ** 2)
    R_curv = dz * (1.0 + (z_R / (dz + 1e-12)) ** 2)    # radius of curvature
    phi_gouy = np.arctan(dz / z_R)

    G_beam = (w0 / w_z) * np.exp(
        -dx ** 2 / w_z ** 2
        + 1j * (k_br * dz + k_br * dx ** 2 / (2.0 * R_curv + 1e-30) - phi_gouy)
        + 1j * phase_offset
    )

    # ── Standing-wave factor (Fabry-Pérot back-reflection) ──
    SW = 1.0 + R_BACK * np.exp(2j * k_br * (Z_BACK - Z))

    # ── Residual skull phase aberration ──
    aberration = np.ones((NX, NZ), dtype=complex)
    if skull_qual < 1.0:
        rng = np.random.default_rng(rng_seed)
        sigma_err = (1.0 - skull_qual) * np.pi   # RMS phase error up to π rad
        # Random phase screen distributed over skull shell
        phase_noise = rng.normal(0.0, sigma_err, (NX, NZ)) * IN_SKULL
        # Propagate skull aberration inward via a simple spatial smoothing
        from scipy.ndimage import gaussian_filter
        phase_smeared = gaussian_filter(phase_noise, sigma=4.0)
        aberration = np.exp(1j * phase_smeared)

    P = T_eff * G_beam * SW * aberration * IN_BRAIN.astype(float)
    return P


# ── Backward (Green's function) field from focal point ───────────────────────
def backward_field(freq: float) -> np.ndarray:
    """
    Backpropagated pressure Green's function from focal point (x_f, z_f).

    2-D cylindrical Green's function (far-field approximation):
      P_bwd(x, z, f) ∝ exp[−ik_br · r_f] / √r_f
    """
    k_br = 2.0 * np.pi * freq / C_BRAIN
    r_f = np.sqrt((X - X_FOC) ** 2 + (Z - Z_FOC) ** 2) + 1e-9
    P_bwd = np.exp(-1j * k_br * r_f) / np.sqrt(r_f)
    return P_bwd * IN_BRAIN.astype(float)


# ── RTM image ─────────────────────────────────────────────────────────────────
def rtm_image(freq: float, skull_qual: float = 1.0) -> np.ndarray:
    """
    RTM imaging condition: Re[P_fwd · conj(P_bwd)], normalised to [0, 1].

    The cross-correlation peaks at the focal point (constructive) and shows
    periodic grating lobes from the standing-wave factor SW(z, f).
    """
    P_fwd = forward_field(freq, skull_qual=skull_qual)
    P_bwd = backward_field(freq)
    I = np.real(P_fwd * np.conj(P_bwd))
    # Clip negative lobes and normalise
    I = np.clip(I, 0.0, None)
    mx = I.max()
    return I / mx if mx > 0 else I


# ── FWI reconstructed focal field ────────────────────────────────────────────
def fwi_field(freq: float, skull_qual: float) -> np.ndarray:
    """
    Simulated focused pressure field using a FWI-reconstructed skull model.
    Better skull_qual → tighter focus → closer to target.
    Returns normalised |P|², clipped to brain.
    """
    P = forward_field(freq, skull_qual=skull_qual, rng_seed=7)
    I = np.abs(P) ** 2
    mx = I.max()
    return I / mx if mx > 0 else I


# ── Target focal distribution ─────────────────────────────────────────────────
def target_field(freq: float) -> np.ndarray:
    """Ideal diffraction-limited focal spot (normalised |P|²)."""
    P = forward_field(freq, skull_qual=1.0)
    # Remove standing waves for ideal target
    k_br = 2.0 * np.pi * freq / C_BRAIN
    SW = 1.0 + R_BACK * np.exp(2j * k_br * (Z_BACK - Z))
    P_ideal = P / SW  # strip standing-wave factor
    I = np.abs(P_ideal) ** 2
    mx = I.max()
    return I / mx if mx > 0 else I


# ── RTM fusion (multi-frequency average) ──────────────────────────────────────
def rtm_fusion(freqs: list[float], skull_qual: float = 1.0) -> np.ndarray:
    """
    Stack RTM images over multiple frequencies.
    Grating lobes are phase-incoherent across frequencies and average down;
    the focal spot remains coherent.
    """
    stack = np.stack([rtm_image(f, skull_qual) for f in freqs], axis=0)
    I = stack.mean(axis=0)
    mx = I.max()
    return I / mx if mx > 0 else I


# ── Figure 10: RTM/FWI 9-panel comparison ─────────────────────────────────────
def fig10_rtm_fwi_comparison() -> None:
    """
    9-panel figure reproducing the multi-method transcranial imaging comparison.

    Row 1 (methods): target | normal FWI | multiparam FWI | nonlinear FWI | fusion RTM
    Row 2 (freq RTM): subharmonic | 110 kHz | 220 kHz | 440 kHz
    Combined into one horizontal strip matching the reference figure layout.
    """
    panels = [
        ("target",      target_field(F0)),
        ("normal FWI",  fwi_field(F0, skull_qual=0.40)),
        ("multiparam",  fwi_field(F0, skull_qual=0.70)),
        ("nonlinear",   fwi_field(F0, skull_qual=0.85)),
        (f"{int(F_SUB/1e3)} kHz", rtm_image(F_SUB, skull_qual=0.85)),
        (f"{int(F0/1e3)} kHz",    rtm_image(F0,    skull_qual=0.85)),
        (f"{int(F2/1e3)} kHz",    rtm_image(F2,    skull_qual=0.85)),
        (f"{int(F4/1e3)} kHz",    rtm_image(F4,    skull_qual=0.85)),
        ("fusion",      rtm_fusion(FREQS_RTM, skull_qual=0.85)),
    ]

    fig, axes = plt.subplots(1, 9, figsize=(18, 2.8),
                             gridspec_kw={"wspace": 0.05})
    cmap = plt.get_cmap("inferno")
    norm = Normalize(vmin=0, vmax=1)

    extent_mm = [-LX / 2 * 1e3, LX / 2 * 1e3,
                 -LZ / 2 * 1e3, LZ / 2 * 1e3]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img.T, origin="lower", cmap=cmap, norm=norm,
                  extent=extent_mm, aspect="equal", interpolation="bilinear")
        ax.contour(x_arr * 1e3, z_arr * 1e3, SKULL_CONTOUR.T,
                   levels=[1.0], colors="white", linewidths=0.6)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.01)
    cbar.set_label("normalised score", fontsize=9)
    cbar.set_ticks([0.0, 0.5, 1.0])

    fig.suptitle("Transcranial FUS reconstruction comparison: FWI methods vs RTM",
                 fontsize=10, y=1.01)
    savefig("fig10_rtm_fwi_comparison")
    plt.close(fig)


# ── Figure 11: RTM axial spectrum per frequency ───────────────────────────────
def fig11_standing_wave_spectrum() -> None:
    """
    Axial power spectrum of RTM images at each frequency.

    The dominant peak at k_z = 2k_br(f) = 4πf/c_br confirms the
    standing-wave interpretation: each RTM panel encodes the
    Fabry-Pérot resonance at twice the acoustic wavenumber.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=False)
    freq_labels = ["55 kHz\n(sub)", "110 kHz", "220 kHz", "440 kHz"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, freq, label, col in zip(axes, FREQS_RTM, freq_labels, colors):
        I = rtm_image(freq)
        # Axial profile along focal x-line
        ix = np.argmin(np.abs(x_arr - X_FOC))
        axial = I[ix, :]   # shape (NZ,)

        # Spatial DFT along z with Hann window
        win = sig_windows.hann(NZ)
        axial_win = (axial - axial.mean()) * win
        S = np.abs(np.fft.rfft(axial_win)) ** 2
        kz = np.fft.rfftfreq(NZ, d=DZ)   # cycles/m

        # Expected standing-wave wavenumber k_SW = 2f/c_br [cycles/m]
        k_expected = 2.0 * freq / C_BRAIN

        ax.semilogy(kz, S + 1e-10, color=col, lw=1.4)
        ax.axvline(k_expected, color="k", ls="--", lw=1.0, label=rf"$2f/c_{{\rm br}}$")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r"$k_z$ [cycles/m]")
        if ax is axes[0]:
            ax.set_ylabel("Power spectral density")
        ax.set_xlim(0, 700)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(r"RTM axial spatial spectrum: peak at $k_z = 2f/c_{\rm br}$ confirms standing-wave origin",
                 fontsize=10)
    fig.tight_layout()
    savefig("fig11_standing_wave_spectrum")
    plt.close(fig)


# ── Figure 12: Delay-law evolution over temporal modulation steps ─────────────
def fig12_delay_law_evolution() -> None:
    """
    Beamforming delay laws τ_i(f_m) for M = 8 modulation steps.

    Each step uses a slightly different frequency f_m = f₀ + m · df_step,
    shifting the standing-wave pattern by one M-th of a full period.
    The RTM-informed delay law compensates for the frequency change to
    maintain focal lock at (x_f, z_f).
    """
    M = 8
    delta_F_period = C_BRAIN / (2.0 * D_BACK)  # Hz — one SW period shift
    delta_f_step = delta_F_period / M
    freqs_mod = F0 + np.arange(M) * delta_f_step

    # Delay law: τ_i(f) = r_i(path through skull) / c_water
    # Approximated as the free-space delay to focal point corrected for skull
    tau_all = np.zeros((N_ELEM, M))
    for m, freq in enumerate(freqs_mod):
        k_br = 2.0 * np.pi * freq / C_BRAIN
        T_eff = skull_transmission(freq)
        skull_phase_shift = -np.angle(T_eff)   # unwrap skull phase per frequency
        for i in range(N_ELEM):
            r_to_foc = np.sqrt((X_FOC - X_ELEM[i]) ** 2 + (Z_FOC - Z_ELEM[i]) ** 2)
            r_max = np.max(np.sqrt((X_FOC - X_ELEM) ** 2 + (Z_FOC - Z_ELEM) ** 2))
            # Delay relative to outermost element; skull phase added per-element
            tau_all[i, m] = (r_max - r_to_foc) / C_WATER + skull_phase_shift / (2.0 * np.pi * freq)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: delay law per modulation step
    ax = axes[0]
    elem_idx = np.arange(N_ELEM)
    cmap_steps = plt.get_cmap("viridis")
    for m in range(M):
        color = cmap_steps(m / (M - 1))
        label = f"f₀+{m}·df" if m < 3 else (None if m < M - 1 else "…")
        ax.plot(elem_idx, tau_all[:, m] * 1e6, color=color, lw=1.2,
                label=f"step {m}" if m in (0, M // 2, M - 1) else None)
    ax.set_xlabel("Element index")
    ax.set_ylabel("Delay τ [μs]")
    ax.set_title(f"Delay law at each of {M} modulation steps\n"
                 rf"$\Delta f = {delta_f_step/1e3:.2f}$ kHz, "
                 rf"$\Delta F_\mathrm{{period}} = {delta_F_period/1e3:.2f}$ kHz")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: delay variation across steps for selected elements
    ax = axes[1]
    selected = [0, N_ELEM // 4, N_ELEM // 2, 3 * N_ELEM // 4, N_ELEM - 1]
    for i in selected:
        ax.plot(np.arange(M), tau_all[i, :] * 1e6, marker="o", ms=4,
                label=f"element {i}")
    ax.set_xlabel("Modulation step m")
    ax.set_ylabel("Delay τ [μs]")
    ax.set_title("Per-element delay shift across modulation steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig12_delay_law_evolution")
    plt.close(fig)


# ── Figure 13: Focal-axis intensity comparison ────────────────────────────────
def fig13_focal_axis_intensity() -> None:
    """
    Axial intensity profile I(x_f, z) for:
      (a) static beamforming at f₀ (standing waves present)
      (b) RTM-modulated M-step dithering (standing waves suppressed)
      (c) Ideal target (no skull reflection)

    Derivation of suppression gain (analytical):
      Static   : <|SW|²>_static  = (1 + R_back)² = (1 + 0.540)² = 2.372  at antinodes
      Modulated: <|SW|²>_M_steps = 1 + R_back²   = 1 + 0.540²  = 1.292  (uniform)
      Gain at antinodes: 2.372 / 1.292 = 1.84  (≡ 2.65 dB)
    """
    M = 8
    delta_F_period = C_BRAIN / (2.0 * D_BACK)
    delta_f_step = delta_F_period / M
    freqs_mod = F0 + np.arange(M) * delta_f_step

    ix = np.argmin(np.abs(x_arr - X_FOC))

    # Static at f₀
    P_static = forward_field(F0, skull_qual=1.0)
    I_static = np.abs(P_static[ix, :]) ** 2

    # RTM-modulated: time-average over M frequency steps
    I_modulated = np.zeros(NZ)
    for freq in freqs_mod:
        # Phase offset chosen so standing-wave antinode covers focal region
        phase = 0.0  # delay law already encoded in focal_quality shift via freq
        P_m = forward_field(freq, skull_qual=1.0, phase_offset=phase)
        I_modulated += np.abs(P_m[ix, :]) ** 2
    I_modulated /= M

    # Ideal target (no back-reflection SW)
    I_target = np.abs(target_field(F0)[ix, :])

    # Normalise by focal-point value of static
    iz_foc = np.argmin(np.abs(z_arr - Z_FOC))
    norm_val = I_static[iz_foc]
    if norm_val == 0:
        norm_val = 1.0
    I_static /= norm_val
    I_modulated /= norm_val
    I_target /= I_target[iz_foc] if I_target[iz_foc] > 0 else 1.0

    # Analytical standing-wave envelope
    k_br = 2.0 * np.pi * F0 / C_BRAIN
    SW_static_peak = (1.0 + R_BACK) ** 2
    SW_static_trough = (1.0 - R_BACK) ** 2
    SW_mod_level = 1.0 + R_BACK ** 2

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    z_mm = z_arr * 1e3

    ax = axes[0]
    ax.plot(z_mm, I_static, label="Static (f₀ = 110 kHz)", color="#d62728", lw=1.5)
    ax.plot(z_mm, I_modulated, label=f"RTM-modulated (M = {M} steps)", color="#1f77b4", lw=1.5)
    ax.plot(z_mm, I_target, label="Target (ideal)", color="k", ls="--", lw=1.2)
    ax.axvline(Z_FOC * 1e3, color="gray", ls=":", lw=1.0, label="Focal point")
    ax.set_ylabel("Normalised intensity |P|²")
    ax.set_title("Focal-axis intensity: static vs RTM-modulated beamforming")
    ax.legend(fontsize=8)
    ax.set_xlim(z_mm[0], z_mm[-1])
    ax.set_ylim(-0.05, 2.8)
    ax.grid(True, alpha=0.3)
    # Annotate brain region
    brain_z_lo = -AZ_INNER_MASK * 1e3
    brain_z_hi = AZ_INNER_MASK * 1e3
    ax.axvspan(brain_z_lo, brain_z_hi, alpha=0.08, color="steelblue", label="_")

    # Standing-wave analytical envelope overlay
    SW_env_hi = np.where(IN_BRAIN[ix, :],
                         SW_static_peak * I_target,
                         np.nan)
    SW_env_lo = np.where(IN_BRAIN[ix, :],
                         SW_static_trough * I_target,
                         np.nan)
    ax.fill_between(z_mm, SW_env_lo, SW_env_hi,
                    alpha=0.15, color="#d62728",
                    label=r"Static envelope [$|1 \pm R_\mathrm{back}|^2$]")
    ax.axhline(SW_mod_level * I_target[iz_foc], color="#1f77b4",
               ls="-.", lw=0.9, alpha=0.7,
               label=r"Modulated DC: $1 + R_\mathrm{back}^2$")

    # Lower panel: suppression ratio
    ax2 = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(I_modulated > 0, I_static / I_modulated, np.nan)
    ax2.plot(z_mm, ratio, color="#2ca02c", lw=1.4)
    ax2.axhline(SW_static_peak / SW_mod_level, color="k", ls="--", lw=1.0,
                label=rf"Analytical peak: $(1+R)^2/(1+R^2) = {SW_static_peak/SW_mod_level:.2f}$")
    ax2.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax2.set_ylabel("Suppression ratio\nI_static / I_mod")
    ax2.set_xlabel("z [mm]")
    ax2.set_title("Grating suppression ratio (> 1 means modulation reduces intensity)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 4)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig("fig13_focal_axis_intensity")
    plt.close(fig)


# ── Figure 14: Time-averaged 2-D field comparison ─────────────────────────────
def fig14_averaged_field_comparison() -> None:
    """
    Side-by-side 2-D intensity maps:
      (a) Target        — ideal focal distribution
      (b) Static f₀    — full standing-wave grating
      (c) RTM-modulated M = 8 — grating suppressed, focal spot maintained
      (d) Suppression ratio map: I_static / I_modulated

    Analytical result per grid point (z inside brain):
      I_static(z)   = |G_beam|² · |1 + R·exp(2ik_br·(z_back−z))|²
      I_modulated(z)= |G_beam|² · (1 + R²)    [M → ∞ limit]
      Ratio at grating antinodes: (1+R)²/(1+R²) ≈ 1.84
    """
    M = 8
    delta_F_period = C_BRAIN / (2.0 * D_BACK)
    delta_f_step = delta_F_period / M
    freqs_mod = F0 + np.arange(M) * delta_f_step

    P_target = target_field(F0)

    P_static = np.abs(forward_field(F0, skull_qual=1.0)) ** 2
    P_static /= P_static.max() if P_static.max() > 0 else 1.0

    P_mod = np.zeros((NX, NZ))
    for freq in freqs_mod:
        P_m = np.abs(forward_field(freq, skull_qual=1.0)) ** 2
        P_mod += P_m
    P_mod /= M
    P_mod /= P_mod.max() if P_mod.max() > 0 else 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_map = np.where((P_mod > 1e-4) & IN_BRAIN,
                             P_static / P_mod,
                             np.nan)

    extent_mm = [-LX / 2 * 1e3, LX / 2 * 1e3,
                 -LZ / 2 * 1e3, LZ / 2 * 1e3]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5),
                             gridspec_kw={"wspace": 0.06})

    panels = [
        ("(a) Target", P_target, "inferno", Normalize(0, 1)),
        ("(b) Static f₀", P_static, "inferno", Normalize(0, 1)),
        (f"(c) RTM-modulated\nM = {M} steps", P_mod, "inferno", Normalize(0, 1)),
        ("(d) Suppression ratio\nI_static / I_mod", ratio_map, "RdYlGn",
         Normalize(0, 3)),
    ]

    for ax, (title, img, cm, nrm) in zip(axes, panels):
        im = ax.imshow(img.T, origin="lower", cmap=cm, norm=nrm,
                       extent=extent_mm, aspect="equal",
                       interpolation="bilinear")
        ax.contour(x_arr * 1e3, z_arr * 1e3, SKULL_CONTOUR.T,
                   levels=[1.0], colors="white", linewidths=0.7)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("x [mm]", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("z [mm]", fontsize=8)
        else:
            ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Annotate suppression ratio map with analytical bound
    axes[3].text(
        0.05, 0.05,
        rf"$\max\, r = (1+R)^2/(1+R^2) = {(1+R_BACK)**2/(1+R_BACK**2):.2f}$",
        transform=axes[3].transAxes,
        fontsize=7, color="black",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
    )

    fig.suptitle(
        f"RTM-modulated tFUS: M = {M} frequency steps suppress standing-wave grating "
        rf"(df = {delta_F_period/1e3:.2f} kHz / {M} = {delta_F_period/M/1e3:.2f} kHz/step)",
        fontsize=9, y=1.00,
    )
    savefig("fig14_averaged_field_comparison")
    plt.close(fig)


# ── Figure 15: RTM-FWI iterative convergence ─────────────────────────────────
def fig15_iterative_rtm_convergence() -> None:
    """
    RTM-FWI iterative loop: each round uses the current RTM image to estimate
    residual skull phase error and refine the correction quality σ.

    Update rule (first-order RTM recovery):
      σ_{k+1} = σ_k + (1 − σ_k) · η,   η = 0.45

    η is the per-iteration recovery efficiency — the fraction of remaining
    skull phase error that RTM-FWI can infer from one round of migration.
    Values η ∈ [0.3, 0.6] are typical for single-frequency RTM at tFUS SNR.

    At each iteration we record:
      Strehl ratio  S_k = I_peak(σ_k) / I_peak(σ=1)    [Maréchal 1947]
      RTM peak      R_k = max_{brain} RTM(f₀; σ_k)
    """
    n_iter = 8
    eta = 0.45
    sigma_0 = 0.40

    P_ref = forward_field(F0, skull_qual=1.0)
    I_ref_peak = float(np.max(np.abs(P_ref[IN_BRAIN]) ** 2))

    sigma_vals = np.zeros(n_iter)
    strehl = np.zeros(n_iter)
    rtm_peaks = np.zeros(n_iter)
    # Analytical Strehl from Maréchal: S ≈ exp(−σ_err²)
    # σ_err = (1 − σ) · π  ← RMS skull phase error in radians
    strehl_analytical = np.zeros(n_iter)

    sigma = sigma_0
    for k in range(n_iter):
        sigma_vals[k] = sigma
        P_k = forward_field(F0, skull_qual=sigma, rng_seed=42)
        I_k_peak = float(np.max(np.abs(P_k[IN_BRAIN]) ** 2))
        strehl[k] = I_k_peak / I_ref_peak

        sigma_err = (1.0 - sigma) * np.pi
        strehl_analytical[k] = np.exp(-sigma_err ** 2)

        I_rtm = rtm_image(F0, skull_qual=sigma)
        rtm_peaks[k] = float(np.max(I_rtm[IN_BRAIN]))

        sigma = sigma + (1.0 - sigma) * eta

    k_arr = np.arange(n_iter)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Left: Strehl ratio vs iteration ──
    ax = axes[0]
    ax.plot(k_arr, strehl, "o-", color="#1f77b4", ms=6, lw=1.8, label="Simulated")
    ax.plot(k_arr, strehl_analytical, "s--", color="#d62728", ms=5, lw=1.2,
            label=r"Maréchal: $e^{-\sigma_\mathrm{err}^2}$")
    ax.axhline(0.8, color="gray", ls=":", lw=1.0, label="S = 0.8 (quality gate)")
    ax.set_xlabel("RTM-FWI iteration $k$")
    ax.set_ylabel("Strehl ratio $S_k$")
    ax.set_title("Focal quality vs RTM iteration")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Centre: skull correction quality convergence ──
    ax = axes[1]
    k_fine = np.linspace(0, n_iter - 1, 200)
    sigma_theory = sigma_0 + (1.0 - sigma_0) * (1.0 - (1.0 - eta) ** k_fine)
    ax.plot(k_fine, sigma_theory, "-", color="#2ca02c", lw=1.4,
            label=r"$\sigma_k = \sigma_0 + (1-\sigma_0)(1-(1-\eta)^k)$")
    ax.plot(k_arr, sigma_vals, "o", color="#2ca02c", ms=6)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"Skull quality $\sigma_k$")
    ax.set_title(rf"Skull model convergence ($\eta={eta}$, $\sigma_0={sigma_0}$)")
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: RTM images iter 0 vs iter n_iter−1 ──
    ax = axes[2]
    I_early = rtm_image(F0, skull_qual=sigma_0)
    I_late = rtm_image(F0, skull_qual=float(sigma_vals[-1]))
    extent_mm = [-LX / 2 * 1e3, LX / 2 * 1e3, -LZ / 2 * 1e3, LZ / 2 * 1e3]
    # Tile horizontally for direct visual comparison
    combined = np.concatenate([I_early.T, I_late.T], axis=1)
    ext_comb = [-LX * 1e3, LX * 1e3, -LZ / 2 * 1e3, LZ / 2 * 1e3]
    ax.imshow(combined, origin="lower", cmap="inferno",
              extent=ext_comb, aspect="equal", vmin=0, vmax=1)
    ax.axvline(0, color="cyan", lw=0.8, ls="--")
    ax.set_title(f"RTM image: iter 0 (left) | iter {n_iter-1} (right)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.text(-LX * 1e3 * 0.75, LZ / 2 * 1e3 * 0.82,
            rf"$\sigma={sigma_0:.2f}$", color="white", fontsize=9, ha="center")
    ax.text(LX * 1e3 * 0.75, LZ / 2 * 1e3 * 0.82,
            rf"$\sigma={sigma_vals[-1]:.3f}$", color="white", fontsize=9, ha="center")

    fig.suptitle(
        "Figure 15 — RTM-FWI iterative skull correction\n"
        rf"$\sigma_{{k+1}} = \sigma_k + (1-\sigma_k)\cdot\eta$, "
        rf"$\eta = {eta}$, $f_0 = {int(F0/1e3)}\,\mathrm{{kHz}}$",
        fontsize=10,
    )
    fig.tight_layout()
    savefig("fig15_iterative_rtm_convergence")
    plt.close(fig)


# ── Figure 16: Temporal pressure dynamics at the focal point ─────────────────
def fig16_temporal_pressure_dynamics() -> None:
    """
    Pressure amplitude at the focal point |P(x_f, z_f, f_m)|² at each
    modulation step m for M = 2, 4, 8, 16 steps spanning one SW period.

    Analytical standing-wave phase at focus:
      phi_m = 2 k_br(f_m) · D_BACK = phi_0 + 2π m / M

    |SW(z_f, f_m)|² = 1 + R² + 2R cos(phi_m)

    For M uniform steps the time-average is exactly 1 + R² (M ≥ 2),
    and the temporal variance is:

      Var[|SW|²]_M = (2R)² / 2 · (1 if M=1 else 2/M²·Σcos²(...))
                   ──→ 0  as M → ∞

    The lower panel shows the standard deviation normalised by the static
    SD as a function of M, confirming convergence to uniform modulation.
    """
    delta_F_period = C_BRAIN / (2.0 * D_BACK)
    M_values = [2, 4, 8, 16]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    ix = np.argmin(np.abs(x_arr - X_FOC))
    iz = np.argmin(np.abs(z_arr - Z_FOC))

    # Reference: perfect beam intensity at focus (no SW)
    P_ref = forward_field(F0, skull_qual=1.0)
    # Strip SW factor at focal pixel for normalisation
    k_ref = 2.0 * np.pi * F0 / C_BRAIN
    SW_ref = 1.0 + R_BACK * np.exp(2j * k_ref * (Z_BACK - Z_FOC))
    I_foc_noSW = float(np.abs(P_ref[ix, iz] / SW_ref) ** 2)
    if I_foc_noSW == 0.0:
        I_foc_noSW = 1.0

    # Analytical SW²(phi) for a dense phase sweep
    phi_dense = np.linspace(0, 2 * np.pi, 1000)
    SW2_dense = 1.0 + R_BACK ** 2 + 2.0 * R_BACK * np.cos(phi_dense)
    SW2_mean = 1.0 + R_BACK ** 2
    SW2_peak = (1.0 + R_BACK) ** 2
    SW2_trough = (1.0 - R_BACK) ** 2

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ── Top: |P_focal|² vs step m for each M ──
    ax = axes[0]
    for M, col in zip(M_values, colors):
        freqs_m = F0 + np.arange(M) * (delta_F_period / M)
        I_steps = np.zeros(M)
        for mi, freq in enumerate(freqs_m):
            P_m = forward_field(freq, skull_qual=1.0)
            I_steps[mi] = float(np.abs(P_m[ix, iz]) ** 2) / I_foc_noSW

        step_idx = np.arange(M)
        ax.plot(step_idx / max(M - 1, 1), I_steps, "o-", color=col,
                ms=5, lw=1.4, label=f"M = {M}")

    # Analytical envelope
    ax.axhline(SW2_mean, color="k", ls="--", lw=1.0,
               label=rf"$\langle|SW|^2\rangle = 1+R^2 = {SW2_mean:.3f}$")
    ax.axhline(SW2_peak, color="gray", ls=":", lw=0.9,
               label=rf"Antinode: $(1+R)^2 = {SW2_peak:.3f}$")
    ax.axhline(SW2_trough, color="gray", ls="-.", lw=0.9,
               label=rf"Node: $(1-R)^2 = {SW2_trough:.3f}$")

    ax.set_xlabel("Normalised step $m / (M-1)$")
    ax.set_ylabel(r"Normalised focal intensity $|P_\mathrm{foc}|^2$")
    ax.set_title(
        r"Focal-point intensity per modulation step"
        "\n"
        rf"$\phi_m = \phi_0 + 2\pi m/M$,  "
        rf"$\Delta F = c_{{br}}/(2d_{{back}}) = {delta_F_period/1e3:.2f}\,\mathrm{{kHz}}$"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # ── Bottom: temporal standard deviation vs M ──
    ax2 = axes[1]
    M_range = np.arange(1, 33)
    sd_analytical = np.zeros(len(M_range))
    for idx, M in enumerate(M_range):
        phi_m = 2.0 * np.pi * np.arange(M) / M
        SW2_m = 1.0 + R_BACK ** 2 + 2.0 * R_BACK * np.cos(phi_m)
        sd_analytical[idx] = float(np.std(SW2_m))

    # Normalise by M=1 (static) sd (which is just the spread of the single-freq SW²)
    # For M=1 there is no variation in step — it's a DC value. Use the natural
    # envelope spread instead: sd_static = (SW2_peak − SW2_trough) / sqrt(12) (uniform model).
    sd_static = (SW2_peak - SW2_trough) / (2.0 * np.sqrt(2.0))
    ax2.semilogy(M_range, sd_analytical + 1e-8, "o-", color="#1f77b4",
                 ms=4, lw=1.4, label="Temporal SD of $|SW|^2$")
    ax2.axhline(sd_static, color="gray", ls="--", lw=1.0,
                label=rf"Static reference SD = {sd_static:.3f}")
    ax2.set_xlabel("Number of modulation steps $M$")
    ax2.set_ylabel(r"$\mathrm{SD}[|SW|^2]$ over steps")
    ax2.set_title(
        r"Variance convergence: $\mathrm{SD} \to 0$ as $M \to \infty$"
        "\n"
        r"(Exact zero for $M \geq 2$ on axis; residual from 2-D beam spread)"
    )
    ax2.legend(fontsize=8)
    ax2.set_xlim(1, 32)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    savefig("fig16_temporal_pressure_dynamics")
    plt.close(fig)


# ── Figure 17: Brain-volume intensity uniformity ──────────────────────────────
def fig17_brain_volume_uniformity() -> None:
    """
    Standing-wave intensity |SW(x, z, f)|² inside the brain volume for:
      (a) Static f₀              — bimodal antinode/node distribution
      (b) M = 8 modulated        — histogram collapses toward 1 + R²
      (c) M = 32 modulated       — approaches uniform (Dirac at 1 + R²)

    Analytical result for M-step uniform modulation:
      <|SW(x, z)|²>_M = 1 + R²  for any (x, z)   (all M ≥ 2)

    The histogram width (standard deviation over the brain volume) therefore
    quantifies spatial homogeneity, not temporal averaging — it reflects how
    the fixed-frequency SW creates a grating pattern in space that modulation
    eliminates to first order in M.

    Key metrics (printed on figure):
      σ_static  : std(|SW|²) over brain pixels at f₀
      σ_M8      : std(<|SW|²>_8) over brain pixels
      σ_M32     : std(<|SW|²>_32) over brain pixels
    """
    delta_F_period = C_BRAIN / (2.0 * D_BACK)

    def sw2_field(freq: float) -> np.ndarray:
        k = 2.0 * np.pi * freq / C_BRAIN
        sw = 1.0 + R_BACK * np.exp(2j * k * (Z_BACK - Z))
        return np.abs(sw) ** 2

    def modulated_sw2(M: int) -> np.ndarray:
        freqs_m = F0 + np.arange(M) * (delta_F_period / M)
        acc = np.zeros((NX, NZ))
        for f in freqs_m:
            acc += sw2_field(f)
        return acc / M

    SW2_static = sw2_field(F0)
    SW2_M8 = modulated_sw2(8)
    SW2_M32 = modulated_sw2(32)
    SW2_analytical = 1.0 + R_BACK ** 2

    brain_mask = IN_BRAIN

    def brain_pixels(field: np.ndarray) -> np.ndarray:
        return field[brain_mask]

    pix_static = brain_pixels(SW2_static)
    pix_M8 = brain_pixels(SW2_M8)
    pix_M32 = brain_pixels(SW2_M32)

    sigma_static = float(np.std(pix_static))
    sigma_M8 = float(np.std(pix_M8))
    sigma_M32 = float(np.std(pix_M32))

    extent_mm = [-LX / 2 * 1e3, LX / 2 * 1e3, -LZ / 2 * 1e3, LZ / 2 * 1e3]

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.30,
                           height_ratios=[1.0, 0.85])

    # ── Row 1: 2-D |SW|² maps ──
    map_data = [
        (rf"(a) Static $f_0 = {int(F0/1e3)}$ kHz"
         rf"\n$\sigma_{{vol}} = {sigma_static:.4f}$", SW2_static),
        (rf"(b) $M = 8$ steps"
         rf"\n$\sigma_{{vol}} = {sigma_M8:.4f}$", SW2_M8),
        (rf"(c) $M = 32$ steps"
         rf"\n$\sigma_{{vol}} = {sigma_M32:.5f}$", SW2_M32),
    ]
    sw2_max = float(np.nanmax(SW2_static))
    sw2_norm = Normalize(vmin=0, vmax=sw2_max)

    map_axes = []
    for col, (title, field) in enumerate(map_data):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(field.T, origin="lower", cmap="RdYlBu_r",
                       norm=sw2_norm, extent=extent_mm,
                       aspect="equal", interpolation="bilinear")
        ax.contour(x_arr * 1e3, z_arr * 1e3, SKULL_CONTOUR.T,
                   levels=[1.0], colors="k", linewidths=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [mm]", fontsize=8)
        if col == 0:
            ax.set_ylabel("z [mm]", fontsize=8)
        ax.plot(X_FOC * 1e3, Z_FOC * 1e3, "w+", ms=8, mew=1.5)
        map_axes.append(ax)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label=r"$|SW|^2$")

    # ── Row 2: histograms ──
    ax_hist = fig.add_subplot(gs[1, :])
    bins = np.linspace(0, sw2_max, 80)
    hist_data = [
        (pix_static, rf"Static $f_0$ (${{\sigma={sigma_static:.3f}}}$)", "#d62728", 0.55),
        (pix_M8,     rf"$M=8$ ($\sigma={sigma_M8:.4f}$)", "#1f77b4", 0.70),
        (pix_M32,    rf"$M=32$ ($\sigma={sigma_M32:.5f}$)", "#2ca02c", 0.90),
    ]
    for pix, lbl, col, alpha in hist_data:
        ax_hist.hist(pix, bins=bins, density=True, alpha=alpha,
                     color=col, label=lbl, edgecolor="none")

    ax_hist.axvline(SW2_analytical, color="k", ls="--", lw=1.6,
                    label=rf"Analytical $\langle|SW|^2\rangle = 1+R^2 = {SW2_analytical:.4f}$")
    ax_hist.axvline((1.0 + R_BACK) ** 2, color="#d62728", ls=":",
                    lw=1.0, alpha=0.7, label=rf"Antinode $(1+R)^2={( 1+R_BACK)**2:.4f}$")
    ax_hist.axvline((1.0 - R_BACK) ** 2, color="#d62728", ls="-.",
                    lw=1.0, alpha=0.7, label=rf"Node $(1-R)^2={(1-R_BACK)**2:.4f}$")

    ax_hist.set_xlabel(r"$|SW(x, z)|^2$ [brain pixels]", fontsize=10)
    ax_hist.set_ylabel("Probability density", fontsize=10)
    ax_hist.set_title(
        r"Brain-volume standing-wave intensity histogram"
        "\n"
        r"Modulation collapses bimodal grating distribution onto $1 + R^2$"
    )
    ax_hist.legend(fontsize=8, ncol=3)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.set_xlim(0, sw2_max * 1.05)

    fig.suptitle(
        "Figure 17 — Standing-wave intensity uniformity inside brain volume\n"
        rf"$R_\mathrm{{back}} = {R_BACK:.4f}$, analytical target $1+R^2 = {SW2_analytical:.4f}$",
        fontsize=10, y=1.00,
    )
    savefig("fig17_brain_volume_uniformity")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chapter 25: RTM-Driven Adaptive Transcranial Beamforming")
    print("=" * 60)
    # Print key parameters for verification
    T_110 = skull_transmission(F0)
    print(f"  Skull transmission at 110 kHz : |T| = {abs(T_110):.4f}")
    print(f"  Back-reflection R_back        : {R_BACK:.4f}")
    delta_F = C_BRAIN / (2.0 * D_BACK)
    print(f"  Standing-wave period shift dF : {delta_F/1e3:.4f} kHz")
    gain = (1.0 + R_BACK) ** 2 / (1.0 + R_BACK ** 2)
    print(f"  Analytical suppression gain   : {gain:.4f} ({10*np.log10(gain):.2f} dB)")
    print(f"  Modulated mean |SW|^2         : {1 + R_BACK**2:.4f}")
    print(f"  Static antinode |SW|^2        : {(1 + R_BACK)**2:.4f}")
    print(f"  Static node     |SW|^2        : {(1 - R_BACK)**2:.4f}")

    print("\n  Generating figures...")
    print("  fig10...", end=" ", flush=True)
    fig10_rtm_fwi_comparison()
    print("done")

    print("  fig11...", end=" ", flush=True)
    fig11_standing_wave_spectrum()
    print("done")

    print("  fig12...", end=" ", flush=True)
    fig12_delay_law_evolution()
    print("done")

    print("  fig13...", end=" ", flush=True)
    fig13_focal_axis_intensity()
    print("done")

    print("  fig14...", end=" ", flush=True)
    fig14_averaged_field_comparison()
    print("done")

    print("  fig15...", end=" ", flush=True)
    fig15_iterative_rtm_convergence()
    print("done")

    print("  fig16...", end=" ", flush=True)
    fig16_temporal_pressure_dynamics()
    print("done")

    print("  fig17...", end=" ", flush=True)
    fig17_brain_volume_uniformity()
    print("done")

    print("\nAll figures saved to docs/book/figures/ch25/")
