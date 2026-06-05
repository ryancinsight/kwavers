"""
Chapter 25: RTM-Driven Adaptive Transcranial Beamforming
=========================================================

Publication-quality figures for docs/book/rtm_adaptive_beamforming.md.

Physical model
--------------
Domain   : 2-D slab cross-section (x = depth, y = lateral)
           125 × 80 mm, 0.5 mm resolution, Ru FDTD via pykwavers.
Array    : 64-element flat aperture (50 mm), at x = 11 mm from left edge.
Skull    : Slab from x ≈ 75 mm to x ≈ 82 mm (7 mm thick).
           c_sk = 2800 m/s, ρ_sk = 1900 kg/m³.
Coupling : Water, c_w = 1500 m/s, ρ_w = 1000 kg/m³.
Focus    : x ≈ 50 mm from left edge (25 mm before skull front face).

Standing-wave geometry
----------------------
Back-reflector: skull front face at x = SKULL_X_START.
Standing-wave region: coupling water between array and skull.
Back-to-focus distance: D_BACK = 25 mm.
Reflection coefficient: R = (Z_sk − Z_w)/(Z_sk + Z_w) ≈ 0.560.

All wave fields computed by the Rust kwavers FDTD solver (pykwavers).
Closed-form formulas are used where they are exact solutions:
  • Transfer-matrix skull transmission (Brekhovskikh 1980).
  • 2-D cylindrical Green's function (Kupradze 1965).
  • Delay-and-sum time delay law (geometry).
  • Standing-wave statistics: ⟨|SW|²⟩_M = 1 + R² (exact, M ≥ 2).

RTM image (frequency domain)
-----------------------------
  I_RTM(x,y,f) = Re[ P_fwd(x,y,f) · P_bwd*(x,y,f) ]

where P_fwd is the FDTD-computed monochromatic pressure field and
P_bwd is the analytical 2-D Green's function from the focal point.

Modulation period and suppression gain
---------------------------------------
  dF_period = c_w / (2 · D_BACK)   [Hz, one full SW period shift]
  Gain_peak = (1 + R)² / (1 + R²)  [linear; at antinode position]

Figures produced
----------------
fig10  Skull correction comparison: uncorrected vs kwavers-optimised
fig11  RTM axial spatial spectrum at four frequencies
fig12  Delay-law evolution over M = 8 temporal modulation steps
fig13  Focal-axis intensity: static vs RTM-modulated (M = 8)
fig14  Time-averaged 2-D field: standing-wave suppression
fig15  Convergence: SWI objective and focal pressure vs iteration
fig16  Focal-point |P|² per modulation step, variance vs M
fig17  Standing-wave intensity histogram: bimodal → Dirac collapse

References
----------
Vignon et al. (2006) J. Acoust. Soc. Am. 120(4):2330
Deffieux & Konofagou (2010) Ultrasound Med. Biol. 36(7):1117
Marquet et al. (2009) J. Acoust. Soc. Am. 125(4):2388
Baysal et al. (1983) Geophysics 48(11):1514
Brekhovskikh (1980) Waves in Layered Media. Academic Press.
"""

from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import pykwavers  # Rust FDTD via PyO3 — all wave physics executes in Rust

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch25")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "lines.linewidth": 1.5,
})


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150,
                    bbox_inches="tight")
    print(f"  saved: figures/ch25/{name}.{{pdf,png}}")


# ── Physical constants (SSOT: kwavers::core::constants::fundamental) ──────────
C_WATER = 1500.0;  C_SKULL = 2800.0;  C_BRAIN = 1540.0
RHO_WATER = 1000.0; RHO_SKULL = 1900.0; RHO_BRAIN = 1040.0
Z_WATER = RHO_WATER * C_WATER   # 1.500 MRayl
Z_SKULL = RHO_SKULL * C_SKULL   # 5.320 MRayl
Z_BRAIN = RHO_BRAIN * C_BRAIN   # 1.601 MRayl
D_SKULL_M = 7e-3
# Coupling-water side: standing waves form in water between array and skull.
R_BACK = pykwavers.reflection_pressure_coeff(Z_WATER, Z_SKULL)  # ≈ 0.560

# ── FDTD domain parameters ────────────────────────────────────────────────────
NX, NY = 250, 160    # 125 mm × 80 mm at 0.5 mm/cell
DX_M   = 5e-4
PML    = 20
CFL    = 0.25

SOURCE_X     = PML + 2                                  # = 22 (array face at 11 mm)
N_ELEM       = 64
ELEM_Y_MIN   = NY // 2 - 50                             # ±25 mm → 50 mm aperture
ELEM_Y_MAX   = NY // 2 + 50
FOCAL_R      = 6

SKULL_X_START = SOURCE_X + round(0.053 / DX_M)         # = 128 (75 mm from left)
SKULL_X_END   = SKULL_X_START + round(D_SKULL_M / DX_M)  # = 142 (82 mm from left)
FOCUS_X       = SOURCE_X + round(0.028 / DX_M)         # =  78 (50 mm from left)
FOCUS_Y       = NY // 2                                  # =  80

# Back-reflector distance (focus → skull front) = (128-78)×0.5mm = 25 mm
D_BACK_M = (SKULL_X_START - FOCUS_X) * DX_M

# Grid coordinate arrays (cell indices)
_xa = np.arange(NX);  _ya = np.arange(NY)

# Frequencies
F_SUB = 55e3; F0 = 110e3; F2 = 220e3; F4 = 440e3
FREQS_RTM = [F_SUB, F0, F2, F4]

# ── Modulation period and SW statistics (all via Rust) ────────────────────────
# dF_period = c / (2 · D_BACK):
dF_PERIOD  = pykwavers.standing_wave_modulation_period_hz(C_WATER, D_BACK_M)
# (1+R)²/(1+R²) — RTM suppression gain:
GAIN_PEAK  = pykwavers.standing_wave_suppression_gain(R_BACK)
# Statistical moments of |1 + R·exp(2ikx)|²:
SW2_MEAN, SW2_PEAK, SW2_TROUGH = pykwavers.standing_wave_intensity_statistics(R_BACK)


# ── Analytical formulas (exact closed-form, not simulation) ───────────────────
def skull_transmission(freq: float) -> complex:
    """
    Transfer-matrix pressure transmission through skull slab (Brekhovskikh 1980).

    Delegates to pykwavers.skull_transfer_matrix_transmission which implements:
    T = 2 / [(1 + Z3/Z1)cos(k2 d) + i(Z3/Z2 + Z2/Z1)sin(k2 d)]

    Args:
        freq: Frequency [Hz].

    Returns:
        Complex pressure transmission coefficient (dimensionless).
    """
    return complex(pykwavers.skull_transfer_matrix_transmission(
        freq, Z_WATER, Z_SKULL, Z_BRAIN, C_SKULL, D_SKULL_M
    ))


def backward_field(freq: float) -> np.ndarray:
    """
    2-D cylindrical Green's function from focal point (FOCUS_X, FOCUS_Y).

    Exact solution of ∇²G + k²G = -δ(r - r_f) in homogeneous medium
    (Kupradze 1965): G(r) ∝ exp(-ik|r - r_f|) / √|r - r_f|,  k = 2πf/c.
    Delegated to pykwavers.backprop_green_function_2d (Rust).

    Returns complex ndarray of shape (NX, NY).
    """
    bwd_re, bwd_im = pykwavers.backprop_green_function_2d(
        _xa * DX_M, _ya * DX_M,
        FOCUS_X * DX_M, FOCUS_Y * DX_M,
        freq, C_WATER,
    )
    return np.asarray(bwd_re) + 1j * np.asarray(bwd_im)


def analytical_sw2_field(freq: float) -> np.ndarray:
    """
    Analytical |SW(x,y,f)|² on the grid.

    Standing-wave intensity |1 + R·exp(2ik_w(SKULL_X_START - x)·DX_M)|²
    computed by pykwavers.standing_wave_field_1d (Rust).
    Valid in the water coupling region (x < SKULL_X_START); zeroed beyond.

    Returns shape (NX, NY).
    """
    # Distance from each cell to the skull front face [m]; positive = before skull.
    x_dist = (SKULL_X_START - _xa) * DX_M
    sw_x = np.asarray(
        pykwavers.standing_wave_field_1d(x_dist, freq, C_WATER, R_BACK)
    )
    sw_x[_xa >= SKULL_X_START] = 0.0
    return np.outer(sw_x, np.ones(NY))   # broadcast to (NX, NY)


# ── FDTD runner (all physics in Rust) ─────────────────────────────────────────
def run_fdtd(freq: float, n_opt_iter: int = 0, n_snapshots: int = 1) -> dict:
    """Run kwavers FDTD at single frequency.  All wave physics in Rust."""
    return pykwavers.run_standing_wave_suppression(
        nx=NX, ny=NY, dx_m=DX_M, frequency_hz=freq, cfl=CFL, pml_cells=PML,
        c_ref_m_s=C_WATER, c_layer_m_s=C_SKULL,
        rho_ref_kg_m3=RHO_WATER, rho_layer_kg_m3=RHO_SKULL,
        layer_x_start=SKULL_X_START, layer_x_end=SKULL_X_END,
        source_x=SOURCE_X, focus_x=FOCUS_X, focus_y=FOCUS_Y,
        n_elements=N_ELEM, element_y_min=ELEM_Y_MIN, element_y_max=ELEM_Y_MAX,
        focal_radius_cells=FOCAL_R,
        burst_cycles=20.0, accum_skip_cycles=8.0, swi_axis_half_width=4,
        n_opt_iter=n_opt_iter,
        swi_weight=0.70, focal_weight=0.30,
        n_snapshots=n_snapshots,
    )


def fdtd_field(r: dict) -> np.ndarray:
    """Extract complex pressure amplitude from FDTD result dict."""
    return np.asarray(r["final_field_re"]) + 1j * np.asarray(r["final_field_im"])


def rtm_image(P_fwd: np.ndarray, freq: float) -> np.ndarray:
    """
    RTM imaging condition: I = Re[P_fwd · P_bwd*], normalised to [0,1].

    P_fwd: FDTD monochromatic pressure field (NX, NY) complex.
    P_bwd: analytical 2-D Green's function from focal point (exact).
    Delegated to pykwavers.rtm_imaging_condition (Rust, zero-lag cross-correlation).
    """
    bwd_re, bwd_im = pykwavers.backprop_green_function_2d(
        _xa * DX_M, _ya * DX_M,
        FOCUS_X * DX_M, FOCUS_Y * DX_M,
        freq, C_WATER,
    )
    return np.asarray(pykwavers.rtm_imaging_condition(
        np.ascontiguousarray(P_fwd.real),
        np.ascontiguousarray(P_fwd.imag),
        np.asarray(bwd_re),
        np.asarray(bwd_im),
        NX, NY,
    ))


# ── Figure helpers ────────────────────────────────────────────────────────────
def _skull_overlay(ax: plt.Axes) -> None:
    """Draw skull slab boundaries on a depth-vs-lateral axes."""
    for xs in (SKULL_X_START * DX_M * 1e3, SKULL_X_END * DX_M * 1e3):
        ax.axvline(xs, color="white", lw=0.8, ls="--")


def _imshow_field(ax: plt.Axes, field: np.ndarray, norm, cmap: str, title: str) -> object:
    x_mm = _xa * DX_M * 1e3;  y_mm = _ya * DX_M * 1e3
    im = ax.pcolormesh(x_mm, y_mm, field.T, cmap=cmap, norm=norm, rasterized=True)
    ax.axvline(SKULL_X_START * DX_M * 1e3, color="w", lw=0.8, ls="--")
    ax.axvline(SKULL_X_END   * DX_M * 1e3, color="w", lw=0.8, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    return im


# ── Figure 10: Skull correction comparison ────────────────────────────────────
def fig10_skull_correction(r_uncorr: dict, r_corr: dict) -> None:
    """
    FDTD fields: uncorrected (n_opt_iter=0) vs kwavers-optimised (n_opt_iter=25).
    RTM images from cross-correlation with analytical backward field at F0.
    """
    P_unc = fdtd_field(r_uncorr);  P_cor = fdtd_field(r_corr)
    I_unc = np.abs(P_unc)**2;       I_cor = np.abs(P_cor)**2
    mx = max(I_unc.max(), I_cor.max()) or 1.0
    I_rtm_unc = rtm_image(P_unc, F0)
    I_rtm_cor = rtm_image(P_cor, F0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    norm1 = Normalize(0, 1);  cmap = "inferno"
    for ax, img, tit in zip(
        axes.flat,
        [I_unc/mx, I_cor/mx, I_rtm_unc, I_rtm_cor],
        ["(a) Uncorrected |P|² (FDTD, n_opt=0)",
         "(b) Corrected   |P|² (FDTD, n_opt=25)",
         "(c) RTM image — uncorrected",
         "(d) RTM image — corrected"],
    ):
        im = _imshow_field(ax, img, norm1, cmap, tit)
        ax.plot(FOCUS_X * DX_M * 1e3, FOCUS_Y * DX_M * 1e3, "w+", ms=10, mew=1.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"Figure 10 — Skull correction comparison (F = {int(F0/1e3)} kHz)\n"
        f"kwavers FDTD: skull slab c={int(C_SKULL)} m/s, "
        f"D_back={D_BACK_M*1e3:.0f} mm, R={R_BACK:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    savefig("fig10_skull_correction")
    plt.close(fig)


# ── Figure 11: RTM axial spatial spectrum ─────────────────────────────────────
def fig11_standing_wave_spectrum(fdtd_by_freq: dict) -> None:
    """Axial power spectrum of RTM image at each frequency; peak at k_x = 2f/c_w."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, freq, col in zip(axes, FREQS_RTM, colors):
        P = fdtd_field(fdtd_by_freq[freq])
        I = rtm_image(P, freq)
        axial = I[:, FOCUS_Y]   # slice at focal y (shape NX)
        # Crop to water coupling region (SOURCE_X to SKULL_X_START)
        lo, hi = SOURCE_X + 2, SKULL_X_START - 2
        axial_crop = axial[lo:hi] - axial[lo:hi].mean()
        win = np.hanning(len(axial_crop))
        S = np.abs(np.fft.rfft(axial_crop * win))**2
        kx = np.fft.rfftfreq(len(axial_crop), d=DX_M)  # cycles/m
        k_sw = pykwavers.standing_wave_spatial_frequency_cycles_m(freq, C_WATER)
        ax.semilogy(kx, S + 1e-12, color=col, lw=1.4)
        ax.axvline(k_sw, color="k", ls="--", lw=1.0,
                   label=rf"$2f/c_w={k_sw:.0f}$ m$^{{-1}}$")
        ax.set_title(f"{freq/1e3:.0f} kHz", fontsize=10)
        ax.set_xlabel(r"$k_x$ [cycles/m]")
        if ax is axes[0]:
            ax.set_ylabel("Power spectral density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(
        r"Figure 11 — RTM axial spectrum: peak at $k_x = 2f/c_w$ (FDTD forward field)",
        fontsize=10)
    fig.tight_layout()
    savefig("fig11_standing_wave_spectrum")
    plt.close(fig)


# ── Figure 12: Delay-law evolution over M modulation steps ───────────────────
def fig12_delay_law_evolution() -> None:
    """
    Time delays τ_i for M = 8 modulation steps spanning one SW period.

    Geometric delay-and-sum law (exact, frequency-independent):
      τ_i = (r_max − r_i) / c,   r_i = dist(elem_i → focus)

    Computed by pykwavers.delay_law_focus_2d (Rust). The geometric time
    delay is identical at all modulation frequencies; temporal modulation
    shifts the phase law but preserves the time-delay geometry.

    Modulation frequencies from pykwavers.temporal_modulation_frequencies.
    """
    M = 8
    freqs_mod = np.asarray(
        pykwavers.temporal_modulation_frequencies(F0, M, C_WATER, D_BACK_M)
    )

    # Element positions in metres (constant x = array face, varying y)
    elem_x_m = np.full(N_ELEM, SOURCE_X * DX_M)
    elem_y_m = np.linspace(ELEM_Y_MIN, ELEM_Y_MAX, N_ELEM) * DX_M

    # Geometric time delays [s] — frequency-independent
    tau = np.asarray(pykwavers.delay_law_focus_2d(
        elem_x_m, elem_y_m, FOCUS_X * DX_M, FOCUS_Y * DX_M, C_WATER
    ))  # shape (N_ELEM,)

    # Replicate for M steps (each step uses same geometric delay)
    tau_all = np.tile(tau[:, np.newaxis], (1, M))  # (N_ELEM, M)

    elem_idx = np.arange(N_ELEM)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cmap_steps = plt.get_cmap("viridis")
    ax = axes[0]
    for m in range(M):
        c = cmap_steps(m / max(M - 1, 1))
        lbl = f"step {m}" if m in (0, M // 2, M - 1) else None
        ax.plot(elem_idx, tau_all[:, m] * 1e6, color=c, lw=1.2, label=lbl)
    ax.set_xlabel("Element index"); ax.set_ylabel("Delay τ [μs]")
    ax.set_title(
        rf"Delay law per modulation step (geometric DAS)"
        rf" — $M={M}$, $\Delta f={dF_PERIOD/M/1e3:.2f}$ kHz/step, "
        rf"$F_0={F0/1e3:.0f}$ kHz"
    )
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in [0, N_ELEM // 4, N_ELEM // 2, 3 * N_ELEM // 4, N_ELEM - 1]:
        ax.plot(np.arange(M), tau_all[i, :] * 1e6, "o-", ms=4, label=f"elem {i}")
    ax.set_xlabel("Step m"); ax.set_ylabel("Delay τ [μs]")
    ax.set_title("Per-element delay shift across modulation steps")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig12_delay_law_evolution")
    plt.close(fig)


# ── Figure 13: Focal-axis intensity: static vs RTM-modulated ─────────────────
def fig13_focal_axis_intensity(r_static: dict, fdtd_mod_steps: list[dict]) -> None:
    """
    Axial intensity |P(x, y_focus)|² from FDTD:
      (a) Static at F0.
      (b) Time-averaged over M = 8 modulation steps.
    Analytical suppression gain overlay.
    """
    M = len(fdtd_mod_steps)
    P_static = fdtd_field(r_static)
    I_static = np.abs(P_static[:, FOCUS_Y])**2

    I_mod = np.zeros(NX)
    for r in fdtd_mod_steps:
        I_mod += np.abs(fdtd_field(r)[:, FOCUS_Y])**2
    I_mod /= M

    norm_val = I_static[FOCUS_X] if I_static[FOCUS_X] > 0 else 1.0
    x_mm = _xa * DX_M * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax = axes[0]
    ax.plot(x_mm, I_static / norm_val, color="#d62728", label=f"Static $f_0={F0/1e3:.0f}$ kHz")
    ax.plot(x_mm, I_mod / norm_val, color="#1f77b4", label=f"RTM-modulated $M={M}$")
    ax.axvline(FOCUS_X * DX_M * 1e3, color="gray", ls=":", lw=0.9, label="Focus")
    ax.axvline(SKULL_X_START * DX_M * 1e3, color="k", ls="--", lw=0.8, label="Skull face")
    ax.axhline(SW2_MEAN, color="#1f77b4", ls="-.", lw=0.8, alpha=0.7,
               label=rf"$1+R^2={SW2_MEAN:.3f}$")
    ax.set_ylabel("Normalised |P|²"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("Focal-axis intensity: FDTD static vs RTM-modulated")

    ax2 = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(I_mod > 0, I_static / I_mod, np.nan)
    ax2.plot(x_mm, ratio, color="#2ca02c", lw=1.4)
    ax2.axhline(GAIN_PEAK, color="k", ls="--", lw=1.0,
                label=rf"$(1+R)^2/(1+R^2)={GAIN_PEAK:.2f}$ ({10*np.log10(GAIN_PEAK):.2f} dB)")
    ax2.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("I_static / I_mod")
    ax2.set_title("Suppression ratio")
    ax2.legend(fontsize=8); ax2.set_ylim(0, 4); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig("fig13_focal_axis_intensity")
    plt.close(fig)


# ── Figure 14: Time-averaged 2-D field comparison ─────────────────────────────
def fig14_averaged_field_comparison(r_static: dict, fdtd_mod_steps: list[dict]) -> None:
    """FDTD 2-D |P|² maps: static, RTM-modulated (M=8), suppression ratio."""
    M = len(fdtd_mod_steps)
    I_s = np.abs(fdtd_field(r_static))**2
    I_m = sum(np.abs(fdtd_field(r))**2 for r in fdtd_mod_steps) / M
    mx = max(I_s.max(), I_m.max()) or 1.0
    ratio = np.where(I_m > 1e-6 * mx, I_s / I_m, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"wspace": 0.07})
    nrm1 = Normalize(0, 1)
    for ax, img, tit in zip(axes,
                             [I_s / mx, I_m / mx, ratio],
                             ["(a) Static $f_0$", f"(b) RTM-modulated $M={M}$",
                              "(c) Suppression ratio"]):
        norm = nrm1 if "ratio" not in tit.lower() else Normalize(0, 3)
        cmap = "inferno" if "ratio" not in tit.lower() else "RdYlGn"
        im = _imshow_field(ax, img, norm, cmap, tit)
        ax.plot(FOCUS_X * DX_M * 1e3, FOCUS_Y * DX_M * 1e3, "w+", ms=8, mew=1.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[2].text(0.05, 0.05,
                 rf"Peak $(1+R)^2/(1+R^2)={GAIN_PEAK:.2f}$",
                 transform=axes[2].transAxes, fontsize=7,
                 bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))
    fig.suptitle(
        f"Figure 14 — 2-D field: static vs RTM-modulated $M={M}$ steps "
        f"(FDTD, $f_0={F0/1e3:.0f}$ kHz)",
        fontsize=10)
    savefig("fig14_averaged_field_comparison")
    plt.close(fig)


# ── Figure 15: Convergence from the optimised FDTD run ────────────────────────
def fig15_convergence(r_opt: dict) -> None:
    """
    SWI objective and normalised focal pressure vs optimisation iteration,
    extracted directly from the kwavers FDTD optimiser history.
    """
    swi  = np.asarray(r_opt["swi_history"])
    fp   = np.asarray(r_opt["focal_pressure_history"])
    obj  = np.asarray(r_opt["objective_history"])
    k    = np.arange(len(obj))
    fp_ref = float(r_opt["focal_pressure_ref_pa"]) or fp.max() or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(k, swi / (swi[0] or 1.0), "o-", color="#d62728", ms=5, lw=1.6,
            label="SWI (normalised to initial)")
    ax.set_xlabel("Iteration $k$"); ax.set_ylabel("Normalised SWI")
    ax.set_title(f"Standing-wave intensity vs iteration ($f_0={F0/1e3:.0f}$ kHz)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(k, fp / fp_ref, "s-", color="#1f77b4", ms=5, lw=1.6,
            label="Focal pressure / reference")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Iteration $k$"); ax.set_ylabel("|P_foc| / P_ref")
    ax.set_title("Focal pressure maintenance during SWI suppression")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Figure 15 — kwavers FDTD optimiser convergence\n"
        f"n_opt_iter=25, swi_weight=0.70, focal_weight=0.30, f={F0/1e3:.0f} kHz",
        fontsize=10)
    fig.tight_layout()
    savefig("fig15_convergence")
    plt.close(fig)


# ── Figure 16: Focal-point |SW|² per modulation step, variance vs M ──────────
def fig16_temporal_pressure_dynamics() -> None:
    """
    Analytical |SW(x_focus, f_m)|² per step m computed by
    pykwavers.standing_wave_field_1d (Rust):
      SW²(x) = |1 + R·exp(2ik·x)|² = 1 + R² + 2R·cos(2kx),  k = 2πf/c

    with x = D_BACK_M (distance from focus to skull front face).

    Lower panel: temporal SD of |SW|² vs M, confirming variance → 0 (M ≥ 2).
    """
    M_vals = [2, 4, 8, 16]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    d_arr = np.array([D_BACK_M])   # single-element array for focal-point query

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax = axes[0]
    for M, col in zip(M_vals, colors):
        freqs_m = np.asarray(
            pykwavers.temporal_modulation_frequencies(F0, M, C_WATER, D_BACK_M)
        )
        I_steps = [
            float(pykwavers.standing_wave_field_1d(d_arr, float(freq), C_WATER, R_BACK)[0])
            for freq in freqs_m
        ]
        ax.plot(np.arange(M) / max(M - 1, 1), I_steps, "o-", color=col,
                ms=5, lw=1.4, label=f"M = {M}")
    ax.axhline(SW2_MEAN,   color="k", ls="--", lw=1.0,
               label=rf"$\langle|SW|^2\rangle={SW2_MEAN:.3f}$")
    ax.axhline(SW2_PEAK,   color="gray", ls=":",  lw=0.9,
               label=rf"Antinode $(1+R)^2={SW2_PEAK:.3f}$")
    ax.axhline(SW2_TROUGH, color="gray", ls="-.", lw=0.9,
               label=rf"Node $(1-R)^2={SW2_TROUGH:.3f}$")
    ax.set_xlabel("Normalised step $m/(M-1)$")
    ax.set_ylabel(r"$|SW(x_\mathrm{foc}, f_m)|^2$")
    ax.set_title(rf"Focal SW² per step — $\Delta F={dF_PERIOD/1e3:.2f}$ kHz, "
                 rf"$R={R_BACK:.3f}$, $D_\mathrm{{back}}={D_BACK_M*1e3:.0f}$ mm")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    M_range = np.arange(1, 33)
    sd = []
    for M in M_range:
        freqs_m = np.asarray(
            pykwavers.temporal_modulation_frequencies(F0, int(M), C_WATER, D_BACK_M)
        )
        sw2_steps = [
            float(pykwavers.standing_wave_field_1d(d_arr, float(f), C_WATER, R_BACK)[0])
            for f in freqs_m
        ]
        sd.append(float(np.std(sw2_steps)))
    ax2.semilogy(M_range, np.array(sd) + 1e-10, "o-", color="#1f77b4", ms=4, lw=1.4)
    ax2.set_xlabel("Number of steps $M$")
    ax2.set_ylabel(r"$\sigma[|SW|^2]$")
    ax2.set_title(r"Variance convergence: $\sigma \to 0$ (exact for $M \geq 2$ on axis)")
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig("fig16_temporal_pressure_dynamics")
    plt.close(fig)


# ── Figure 17: Brain-volume |SW|² histogram ───────────────────────────────────
def fig17_brain_volume_uniformity() -> None:
    """
    Analytical standing-wave intensity |SW(x)|² in coupling water (x < skull)
    computed by pykwavers.standing_wave_field_1d (Rust).

    Modulation over M steps collapses bimodal → uniform at 1+R² (exact M ≥ 2).
    """
    sw2_static = analytical_sw2_field(F0)

    def modulated_sw2(M: int) -> np.ndarray:
        freqs_m = pykwavers.temporal_modulation_frequencies(F0, M, C_WATER, D_BACK_M)
        return sum(analytical_sw2_field(f) for f in freqs_m) / M

    sw2_M8  = modulated_sw2(8)
    sw2_M32 = modulated_sw2(32)
    mask = _xa < SKULL_X_START   # water coupling region (rows)

    def _pix(field: np.ndarray) -> np.ndarray:
        return field[mask, :].ravel()

    px_s, px_8, px_32 = _pix(sw2_static), _pix(sw2_M8), _pix(sw2_M32)
    sw2_max = float(np.nanmax(sw2_static))
    norm_map = Normalize(0, sw2_max)
    x_mm = _xa * DX_M * 1e3;  y_mm = _ya * DX_M * 1e3

    fig = plt.figure(figsize=(15, 9))
    import matplotlib.gridspec as gs_
    grd = gs_.GridSpec(2, 3, hspace=0.40, wspace=0.30, height_ratios=[1.0, 0.85])

    for col, (title, field, sx) in enumerate([
        (rf"(a) Static $f_0$, $\sigma={np.std(px_s):.4f}$", sw2_static, np.std(px_s)),
        (rf"(b) $M=8$, $\sigma={np.std(px_8):.5f}$",       sw2_M8,     np.std(px_8)),
        (rf"(c) $M=32$, $\sigma={np.std(px_32):.6f}$",      sw2_M32,    np.std(px_32)),
    ]):
        ax = fig.add_subplot(grd[0, col])
        ax.pcolormesh(x_mm, y_mm, field.T, cmap="RdYlBu_r", norm=norm_map,
                      rasterized=True)
        ax.axvline(SKULL_X_START * DX_M * 1e3, color="k", lw=0.9, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.plot(FOCUS_X * DX_M * 1e3, FOCUS_Y * DX_M * 1e3, "w+", ms=8, mew=1.5)

    ax_h = fig.add_subplot(grd[1, :])
    bins = np.linspace(0, sw2_max, 80)
    for pix, lbl, col, alpha in [
        (px_s,  rf"Static $f_0$ ($\sigma={np.std(px_s):.3f}$)",  "#d62728", 0.55),
        (px_8,  rf"$M=8$  ($\sigma={np.std(px_8):.4f}$)",         "#1f77b4", 0.70),
        (px_32, rf"$M=32$ ($\sigma={np.std(px_32):.5f}$)",        "#2ca02c", 0.90),
    ]:
        ax_h.hist(pix, bins=bins, density=True, alpha=alpha, color=col,
                  label=lbl, edgecolor="none")
    ax_h.axvline(SW2_MEAN,   color="k",       ls="--", lw=1.6,
                 label=rf"$1+R^2={SW2_MEAN:.4f}$")
    ax_h.axvline(SW2_PEAK,   color="#d62728", ls=":",  lw=1.0, alpha=0.7,
                 label=rf"$(1+R)^2={SW2_PEAK:.4f}$")
    ax_h.axvline(SW2_TROUGH, color="#d62728", ls="-.", lw=1.0, alpha=0.7,
                 label=rf"$(1-R)^2={SW2_TROUGH:.4f}$")
    ax_h.set_xlabel(r"$|SW(x,y)|^2$"); ax_h.set_ylabel("Probability density")
    ax_h.set_title(r"SW intensity histogram — modulation collapses bimodal → $1+R^2$")
    ax_h.legend(fontsize=8, ncol=3); ax_h.grid(True, alpha=0.3)
    fig.suptitle(
        f"Figure 17 — Standing-wave intensity uniformity\n"
        rf"$R={R_BACK:.4f}$, $1+R^2={SW2_MEAN:.4f}$, $D_\mathrm{{back}}={D_BACK_M*1e3:.0f}$ mm",
        fontsize=10, y=1.00)
    savefig("fig17_brain_volume_uniformity")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chapter 25: RTM-Driven Adaptive Transcranial Beamforming")
    print("=" * 60)
    print(f"  R_back           : {R_BACK:.4f}")
    print(f"  D_back           : {D_BACK_M*1e3:.1f} mm")
    print(f"  dF_period        : {dF_PERIOD/1e3:.3f} kHz")
    print(f"  Suppression gain : {GAIN_PEAK:.4f}  ({10*np.log10(GAIN_PEAK):.2f} dB)")
    print(f"  SW2_mean (mod)   : {SW2_MEAN:.4f}")
    print(f"  SW2_peak (static): {SW2_PEAK:.4f}")

    # ── Run all FDTD simulations (Rust, release build) ────────────────────────
    print("\n  Running FDTD simulations (Rust)...")
    M_STEPS = 8
    freqs_mod = list(pykwavers.temporal_modulation_frequencies(F0, M_STEPS, C_WATER, D_BACK_M))

    print("  [1/4] Static runs at 4 RTM frequencies (n_opt=0)...")
    fdtd_by_freq = {f: run_fdtd(f, n_opt_iter=0) for f in FREQS_RTM}

    print("  [2/4] Modulation steps M=8 (n_opt=0)...")
    fdtd_mod_steps: list[dict] = []
    for freq in freqs_mod:
        r = fdtd_by_freq.get(freq) or run_fdtd(freq, n_opt_iter=0)
        fdtd_mod_steps.append(r)

    print("  [3/4] Optimised run (n_opt=25, F0)...")
    r_opt = run_fdtd(F0, n_opt_iter=25, n_snapshots=3)

    print("  [4/4] Uncorrected baseline (n_opt=0, F0)...")
    r_unc = fdtd_by_freq[F0]   # already computed above

    r_static_f0 = fdtd_by_freq[F0]

    # Optional: CT-conditioned benchmark (requires CH25_CT_NIFTI env var).
    ct_path = os.environ.get("CH25_CT_NIFTI", "")
    if ct_path and os.path.isfile(ct_path):
        print(f"  CT NIfTI found: {ct_path}")
        print("  Running skull-adaptive CT benchmark...")
        ct_result = pykwavers.run_transcranial_skull_adaptive_benchmark_from_ritk_ct(
            ct_nifti_path=ct_path,
            grid_size=64,
            element_count=512,
            frequency_hz=F0,
            brain_sound_speed=C_BRAIN,
            skull_sound_speed=C_SKULL,
        )
        print(f"  CT benchmark: relative_l2={ct_result['metrics']['relative_l2']:.4f}, "
              f"peak_pa={ct_result['metrics']['candidate_peak_pa']:.1f} Pa")
    else:
        if ct_path:
            print(f"  Warning: CH25_CT_NIFTI={ct_path!r} not found; CT panel skipped.")
        else:
            print("  CT benchmark skipped (set CH25_CT_NIFTI=<path> to enable).")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\n  Generating figures...")

    print("  fig10...", end=" ", flush=True)
    fig10_skull_correction(r_unc, r_opt)
    print("done")

    print("  fig11...", end=" ", flush=True)
    fig11_standing_wave_spectrum(fdtd_by_freq)
    print("done")

    print("  fig12...", end=" ", flush=True)
    fig12_delay_law_evolution()
    print("done")

    print("  fig13...", end=" ", flush=True)
    fig13_focal_axis_intensity(r_static_f0, fdtd_mod_steps)
    print("done")

    print("  fig14...", end=" ", flush=True)
    fig14_averaged_field_comparison(r_static_f0, fdtd_mod_steps)
    print("done")

    print("  fig15...", end=" ", flush=True)
    fig15_convergence(r_opt)
    print("done")

    print("  fig16...", end=" ", flush=True)
    fig16_temporal_pressure_dynamics()
    print("done")

    print("  fig17...", end=" ", flush=True)
    fig17_brain_volume_uniformity()
    print("done")

    print("\nAll figures written to:", OUT_DIR)
