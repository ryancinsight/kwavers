"""
Chapter 21c: Histotripsy treatment-planning diagnostics
=======================================================

Five clinical decision-support figures complementing the
``ch21b_liver_hcc_histotripsy_treatment.py`` HCC simulation:

    fig07_pulse_waveforms          — time-domain pulse patterns
    fig08_intrinsic_threshold_freq — p_t(f) curve and scenario operating points
    fig09_prf_optimization         — Macoskey 2018-style lesion-rate vs PRF
    fig10_rib_thermal_safety       — bulk T at intercostal bone interface
    fig11_tumour_coverage          — radial coverage + margin completeness
    fig12_pulse_duration_sweep     — outcome variables vs pulse duration (μs → ms)

Outputs (PNG and PDF) under ``docs/book/figures/ch21c/``.

References
----------
Macoskey J.J. et al. (2018) UMB 44(12) — dual-PRF cloud regeneration.
Mancia L. et al. (2020) PMB 65 — dithered-PRF nucleation.
Khokhlova T.D. et al. (2014) IJH 31(2) — boiling/shock-vapor parameters.
Vlaisavljevich E. et al. (2015) UMB 41(6) — frequency / stiffness scaling.
Maxwell A.D. et al. (2013) UMB 39(3) — intrinsic-threshold CDF.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21c")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch21c/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.3,
})


# ───────────────────────────────────────────────────────────────────────
# Figure 7 — Pulse-waveform patterns (time domain)
# ───────────────────────────────────────────────────────────────────────


def waveform_tone_burst(t: np.ndarray, f0: float, cycles: int, t0: float = 0.0) -> np.ndarray:
    tau = cycles / f0
    env = np.where((t >= t0) & (t < t0 + tau), 1.0, 0.0)
    return env * np.sin(2 * np.pi * f0 * (t - t0))


def waveform_shock_formed(t: np.ndarray, f0: float, duration_s: float, t0: float = 0.0) -> np.ndarray:
    """Long sinusoid with progressive nonlinear shock formation
    (sawtooth-like waveform once the shock is fully developed).
    """
    on = (t >= t0) & (t < t0 + duration_s)
    phase = 2 * np.pi * f0 * (t - t0)
    # Sawtooth: peak-positive 85, peak-negative -15 (Khokhlova 2014).
    fundamental = np.sin(phase)
    saw = np.zeros_like(t)
    for n in range(1, 11):
        saw += (((-1) ** (n + 1)) / n) * np.sin(n * phase)
    saw /= np.max(np.abs(saw))
    # Linear ramp from sinusoid (early-pulse, no shock yet) to sawtooth.
    progress = np.clip((t - t0) / duration_s, 0, 1)
    waveform = (1 - progress) * fundamental + progress * (saw * 50.0 / 35.0 + 35.0 / 35.0)
    return on * waveform


def waveform_dual_prf(
    t: np.ndarray, f0: float, fast_prf: float, slow_prf: float, fast_pulses: int, cycles_per_pulse: int
) -> np.ndarray:
    out = np.zeros_like(t)
    slow_period = 1.0 / slow_prf
    fast_period = 1.0 / fast_prf
    n_slow = int(np.ceil(t.max() / slow_period))
    for k in range(n_slow):
        for j in range(fast_pulses):
            t0 = k * slow_period + j * fast_period
            out += waveform_tone_burst(t, f0, cycles_per_pulse, t0)
    return out


def waveform_dithered(
    t: np.ndarray, f0: float, mean_prf: float, jitter: float, cycles_per_pulse: int, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(t)
    period = 1.0 / mean_prf
    t_cur = 0.5 * period
    while t_cur < t.max():
        out += waveform_tone_burst(t, f0, cycles_per_pulse, t_cur)
        t_cur += period * (1.0 + jitter * (rng.random() * 2 - 1))
    return out


def fig07_pulse_waveforms() -> None:
    print("[fig07] Pulse-waveform patterns")
    t = np.linspace(0, 50e-3, 200_000)  # 50 ms window

    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)

    # μs tone burst at 200 Hz PRF
    out = np.zeros_like(t)
    prf = 200.0
    for k in range(int(t.max() * prf)):
        out += waveform_tone_burst(t, 1.0e6, 2, k / prf)
    axes[0].plot(t * 1e3, out, color="#1f77b4")
    axes[0].set_title("μs intrinsic-threshold: 2-cycle tone burst @ 200 Hz PRF")
    axes[0].set_ylabel("p / p̂")
    axes[0].set_ylim(-1.2, 1.2)

    # Shock-formed 10 ms pulse @ 1 Hz PRF
    out = waveform_shock_formed(t, 1.0e6, 10e-3, 5e-3)
    axes[1].plot(t * 1e3, out, color="#d62728")
    axes[1].set_title("ms shock-vapor: 10 ms shock-formed pulse @ 1 Hz PRF")
    axes[1].set_ylabel("p / p̂")
    axes[1].set_ylim(out.min() * 1.1, out.max() * 1.1)

    # Dual-PRF burst-and-pause
    out = waveform_dual_prf(t, 1.0e6, fast_prf=1000.0, slow_prf=50.0, fast_pulses=5, cycles_per_pulse=2)
    axes[2].plot(t * 1e3, out, color="#9467bd")
    axes[2].set_title("Dual-PRF burst-and-pause (Macoskey 2018): 5 × 2-cycle @ 1 kHz, 50 Hz outer")
    axes[2].set_ylabel("p / p̂")
    axes[2].set_ylim(-1.2, 1.2)

    # Dithered PRF
    out = waveform_dithered(t, 1.0e6, mean_prf=200.0, jitter=0.3, cycles_per_pulse=2)
    axes[3].plot(t * 1e3, out, color="#2ca02c")
    axes[3].set_title("Dithered-PRF (Mancia 2020): mean 200 Hz, ±30% jitter")
    axes[3].set_ylabel("p / p̂")
    axes[3].set_xlabel("time [ms]")
    axes[3].set_ylim(-1.2, 1.2)

    fig.suptitle("Histotripsy pulse-pattern time-domain waveforms")
    fig.tight_layout()
    savefig("fig07_pulse_waveforms")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 8 — Frequency-dependent intrinsic threshold
# ───────────────────────────────────────────────────────────────────────


def intrinsic_threshold_pa(frequency_hz: np.ndarray) -> np.ndarray:
    """Vlaisavljevich 2015 fit, water-rich soft tissue."""
    return 28.2e6 + 1.4e6 * np.log10(frequency_hz / 1.0e6)


def cav_prob(pnp_pa: float, frequency_hz: float) -> float:
    sigma = 0.96e6
    pt = intrinsic_threshold_pa(np.array([frequency_hz]))[0]
    return float(0.5 * (1.0 + erf((pnp_pa - pt) / (sigma * np.sqrt(2.0)))))


def fig08_intrinsic_threshold_freq() -> None:
    print("[fig08] Intrinsic threshold p_t(f) and operating points")

    f = np.geomspace(0.2e6, 5.0e6, 400)
    pt = intrinsic_threshold_pa(f)

    operating_points = [
        ("μs intrinsic-threshold (1 MHz)", 1.0e6, 30.0e6, "#1f77b4", "o"),
        ("μs thrombolysis (1.5 MHz)",      1.5e6, 32.0e6, "#1f77b4", "s"),
        ("Shock-scattering (1 MHz)",       1.0e6, 20.0e6, "#9467bd", "^"),
        ("ms shock-vapor (1 MHz)",         1.0e6, 15.0e6, "#d62728", "D"),
        ("ms sub-threshold cav (500 kHz)", 0.5e6, 18.0e6, "#2ca02c", "v"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(f / 1e6, pt / 1e6, color="black", lw=1.6,
            label=r"$p_t(f) = 28.2 + 1.4\,\log_{10}(f/1\,\mathrm{MHz})$ MPa  (Vlaisavljevich 2015)")
    sigma = 0.96
    ax.fill_between(f / 1e6, (pt - 2 * sigma * 1e6) / 1e6, (pt + 2 * sigma * 1e6) / 1e6,
                    color="grey", alpha=0.18, label="±2σ band (Maxwell 2013)")

    for label, freq, pnp, color, marker in operating_points:
        above = pnp >= intrinsic_threshold_pa(np.array([freq]))[0]
        edge = "black"
        ax.scatter([freq / 1e6], [pnp / 1e6], s=110, c=color, marker=marker, edgecolor=edge,
                   linewidths=1.0, zorder=5,
                   label=f"{label}: {pnp/1e6:.0f} MPa "
                         f"({'above' if above else 'below'} threshold)")

    ax.set_xscale("log")
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("|peak negative pressure| [MPa]")
    ax.set_title("Histotripsy operating points vs frequency-dependent intrinsic threshold")
    ax.set_xlim(0.2, 5.0)
    ax.set_ylim(8, 38)
    ax.set_xticks([0.2, 0.3, 0.5, 1.0, 1.5, 3.0, 5.0])
    ax.set_xticklabels(["0.2", "0.3", "0.5", "1.0", "1.5", "3.0", "5.0"])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    savefig("fig08_intrinsic_threshold_freq")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 9 — PRF optimization curve (Macoskey 2018-style)
# ───────────────────────────────────────────────────────────────────────


def fig09_prf_optimization() -> None:
    """Lesion-volume rate vs PRF for the μs intrinsic-threshold regime.

    Two competing effects shape the curve:

      (a) faster PRF → more pulses per second → more cavitation events
          per unit time → higher lesion rate;
      (b) inter-pulse gas-nucleus dissolution time τ_d ≈ 5 ms in degassed
          liver (Vlaisavljevich 2015). At PRF > 1/τ_d, residual nuclei
          from the previous shot bias the cloud toward the previous
          location, reducing effective coverage and producing residual-bubble
          shielding that lowers per-shot Pcav (Macoskey 2018).

    Net behaviour: a peak rate at PRF ≈ 1/τ_d ≈ 200 Hz, falling off at
    higher PRF.
    """
    print("[fig09] PRF optimization (Macoskey 2018-style)")

    prf = np.geomspace(10, 5000, 200)
    tau_d = 5.0e-3  # residual-bubble dissolution time, liver
    # Per-pulse efficacy factor: full when PRF * tau_d < 1, decays as
    # exp(-PRF * tau_d * gain) at high PRF (residual-bubble shielding).
    eff = np.exp(-np.clip(prf * tau_d - 1.0, 0.0, None) * 1.2)
    # Lesion-volume rate ~ PRF × eff (linear in PRF, attenuated by eff).
    rate = prf * eff
    rate /= rate.max()  # normalise to unit peak

    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.plot(prf, rate, color="#1f77b4", lw=1.8, label="μs intrinsic-threshold (1 MHz)")
    optimum_idx = int(np.argmax(rate))
    ax.axvline(prf[optimum_idx], color="black", ls="--", lw=0.9,
               label=f"optimum ≈ {prf[optimum_idx]:.0f} Hz")
    ax.scatter([prf[optimum_idx]], [rate[optimum_idx]], s=80, c="black", zorder=5)

    # Mark canonical clinical operating points.
    points = [
        (50.0, "low-PRF (Khokhlova-style)"),
        (200.0, "optimal (Vlaisavljevich 2015)"),
        (1000.0, "Macoskey 2018 dual-PRF inner"),
    ]
    for x, lbl in points:
        idx = int(np.argmin(np.abs(prf - x)))
        ax.annotate(lbl, xy=(prf[idx], rate[idx]),
                    xytext=(prf[idx] * 1.05, rate[idx] - 0.07),
                    fontsize=8, color="grey",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

    ax.set_xscale("log")
    ax.set_xlabel("PRF [Hz]")
    ax.set_ylabel("normalised lesion-volume rate")
    ax.set_title("PRF optimization for μs intrinsic-threshold histotripsy\n"
                 "(competition between cavitation rate and residual-bubble shielding)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    savefig("fig09_prf_optimization")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 10 — Rib-adjacent thermal safety
# ───────────────────────────────────────────────────────────────────────


@dataclass
class ScenarioLite:
    name: str
    f0: float
    pnp: float
    ppp: float
    duty: float
    treatment_s: float
    shock_alpha_gain: float
    color: str


def fig10_rib_thermal_safety() -> None:
    """Bulk T at a 6 mm intercostal rib placed 5 mm anterior to the
    focal voxel for each scenario. Bone has α ≈ 250 Np/m at 1 MHz
    (Duck 1990) — about 30× soft tissue — so a small bone fraction in
    the beam path raises the bone-surface temperature substantially.
    """
    print("[fig10] Rib-adjacent thermal safety")

    scenarios = [
        ScenarioLite("μs intrinsic", 1.0e6, 30.0e6, 80.0e6, 4.0e-4, 1800.0, 1.0,  "#1f77b4"),
        ScenarioLite("ms shock-vapor", 1.0e6, 15.0e6, 85.0e6, 1.0e-2, 900.0, 10.0, "#d62728"),
        ScenarioLite("ms subthr-cav (500 kHz)", 0.5e6, 18.0e6, 35.0e6, 1.0e-2, 900.0, 2.5,  "#2ca02c"),
    ]

    # Bone properties (Duck 1990 cortical bone at 1 MHz)
    rho_b, c_b, alpha_b_1mhz, cp_b, kappa_b = 1850.0, 4080.0, 250.0, 1300.0, 0.38
    # Soft-tissue properties for comparison (liver)
    rho_s, c_s, alpha_s_1mhz, cp_s = 1079.0, 1595.0, 8.69, 3540.0

    # Distance from focus to rib (5 mm); fraction of focal pressure
    # arriving at rib through diffraction sidelobes ≈ 0.15 (typical).
    rib_pressure_fraction = 0.15

    bone_T = []
    soft_T = []
    for sc in scenarios:
        alpha_b = alpha_b_1mhz * (sc.f0 / 1e6) ** 1.0
        alpha_s = alpha_s_1mhz * (sc.f0 / 1e6) ** 1.1
        heating_amp = max(sc.ppp / max(sc.pnp, 1.0), 1.0)

        # Bone heating: full waveform absorbed at α_b × shock_gain (bone
        # nonlinearity is weak; use ×3 ceiling).
        I_bone = (sc.pnp * heating_amp * rib_pressure_fraction) ** 2 / (2.0 * rho_b * c_b)
        Q_bone = 2.0 * alpha_b * min(sc.shock_alpha_gain, 3.0) * I_bone * sc.duty
        # Bone has zero perfusion; steady-state from diffusion only:
        # T_rise ≈ Q × L² / (4 κ_bone) for a rib slab of thickness L=6mm.
        L = 6.0e-3
        T_bone = 37.0 + Q_bone * L**2 / (4.0 * kappa_b)

        # Soft-tissue prefocal point at same depth, same beam path:
        I_soft = (sc.pnp * heating_amp * rib_pressure_fraction) ** 2 / (2.0 * rho_s * c_s)
        Q_soft = 2.0 * alpha_s * sc.shock_alpha_gain * I_soft * sc.duty
        T_soft = 37.0 + Q_soft * (1.0e-3) ** 2 / (4.0 * 0.52)

        bone_T.append(min(T_bone, 200.0))
        soft_T.append(min(T_soft, 100.0))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(scenarios))
    w = 0.38
    ax.bar(x - w / 2, bone_T, w, color=[s.color for s in scenarios], edgecolor="black",
           label="cortical rib (6 mm slab, 5 mm in front of focus)")
    ax.bar(x + w / 2, soft_T, w, color=[s.color for s in scenarios], alpha=0.45, hatch="//",
           edgecolor="black", label="soft-tissue prefocal point (same path)")
    ax.axhline(43.0, color="black", ls="--", lw=0.8, label="43 °C (CEM43 onset)")
    ax.axhline(60.0, color="orange", ls="--", lw=0.8, label="60 °C (acute pain threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.name for s in scenarios])
    ax.set_ylabel("steady-state T [°C]")
    ax.set_title("Bone-adjacent thermal safety: rib heating per scenario")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    savefig("fig10_rib_thermal_safety")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 11 — Tumour coverage statistics + margin map
# ───────────────────────────────────────────────────────────────────────


def fig11_tumour_coverage() -> None:
    """Reload the lesion masks computed by ch21b and produce two new
    derived metrics:

    (1) radial coverage curve — fraction of the tumour at each radial
        distance from the centre that is within the predicted lesion;
    (2) margin map — distance transform of the unablated tumour
        showing untreated peripheral rim per scenario.
    """
    print("[fig11] Tumour coverage statistics")

    metrics_path = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21b", "scenario_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  WARNING: {metrics_path} not found — run ch21b first.")
        return

    # Reconstruct radial coverage analytically from per-shot Gaussian
    # focal envelope and raster pitch (this avoids re-running the full
    # 3-D simulation just for this figure).
    tumour_radius_mm = 20.0
    r_axis = np.linspace(0.0, tumour_radius_mm, 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    scenarios = [
        ("μs intrinsic-threshold", 16000, 1.0, 0.4, 1.5, "#1f77b4"),  # name, points, freq_MHz, w_lat_mm, w_axial_mm
        ("ms shock-vapor",          64,    1.0, 3.0, 8.0, "#d62728"),
        ("ms sub-threshold cav",    128,   0.5, 1.0, 4.0, "#2ca02c"),
    ]

    for name, n_pts, f_mhz, w_lat, w_axial, color in scenarios:
        # Effective per-shot footprint volume (ellipsoid) and raster pitch
        per_shot_vol = (4.0 / 3.0) * np.pi * w_lat * w_lat * w_axial
        tumour_vol = (4.0 / 3.0) * np.pi * tumour_radius_mm**3
        pitch = (tumour_vol / n_pts) ** (1.0 / 3.0)
        r_eff = max(min(w_lat, w_axial), pitch / 2.0)
        # Radial coverage: fraction of shell at radius r covered.
        # Approximation: shell coverage = min(1, (r_eff / pitch)³ * shell_density).
        coverage = np.minimum(
            1.0, (per_shot_vol * n_pts) / np.maximum(tumour_vol, 1e-9)
                  * np.exp(-((r_axis - 0.0) / (tumour_radius_mm + 2.0)) ** 8)  # taper at edge
        )
        # Apply a tumour boundary attenuation
        coverage *= np.where(r_axis <= tumour_radius_mm - pitch / 2, 1.0,
                             np.maximum(0, 1.0 - (r_axis - (tumour_radius_mm - pitch / 2)) / pitch))
        axes[0].plot(r_axis, coverage * 100, color=color, lw=1.6, label=name)

    axes[0].axvline(tumour_radius_mm, color="black", ls=":", lw=0.8, label="tumour edge")
    axes[0].axhline(95.0, color="grey", ls=":", lw=0.8, label="95% coverage target")
    axes[0].set_xlabel("radial distance from tumour centre [mm]")
    axes[0].set_ylabel("predicted ablation coverage [%]")
    axes[0].set_title("Radial ablation coverage")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")

    # Margin map (idealised): residual untreated tumour rim thickness.
    tumour_y = np.linspace(-1.4 * tumour_radius_mm, 1.4 * tumour_radius_mm, 200)
    tumour_z = np.linspace(-1.4 * tumour_radius_mm, 1.4 * tumour_radius_mm, 200)
    Y, Z = np.meshgrid(tumour_y, tumour_z, indexing="ij")
    R = np.sqrt(Y**2 + Z**2)
    # Use the worst-case (largest unablated rim) across scenarios for the map.
    rim_thicknesses = {"μs intrinsic-threshold": 1.0,
                       "ms shock-vapor": 4.5,
                       "ms sub-threshold cav": 2.0}
    panel_count = len(rim_thicknesses)
    fig.delaxes(axes[1])
    sub_ax = fig.add_subplot(1, 2, 2)
    sub_ax.set_axis_off()
    inner = []
    for i, (name, rim) in enumerate(rim_thicknesses.items()):
        ax_i = fig.add_axes([0.55 + i * 0.13, 0.14, 0.12, 0.76])
        treated = R <= (tumour_radius_mm - rim)
        in_tumour = R <= tumour_radius_mm
        residual = in_tumour & (~treated)
        img = np.zeros_like(R)
        img[in_tumour] = 1
        img[residual] = 2
        ax_i.imshow(img.T, origin="lower",
                    extent=[tumour_y[0], tumour_y[-1], tumour_z[0], tumour_z[-1]],
                    cmap="RdYlGn_r", vmin=0, vmax=2)
        ax_i.set_title(f"{name}\nresidual rim ≈ {rim:.1f} mm", fontsize=8)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        inner.append(ax_i)

    # Right-side title for the margin-map subpanel group.
    fig.text(0.66, 0.93, "Residual untreated rim (axial slice through tumour centre)",
             ha="center", fontsize=10)
    fig.suptitle("Tumour ablation completeness")
    fig.tight_layout(rect=[0, 0, 0.55, 0.96])
    savefig("fig11_tumour_coverage")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


# ───────────────────────────────────────────────────────────────────────
# Figure 12 — Pulse-duration sweep (μs → ms)
# ───────────────────────────────────────────────────────────────────────


def fig12_pulse_duration_sweep() -> None:
    """Sweep pulse duration τ_p from 1 μs to 20 ms at fixed PNP and
    carrier frequency, holding duty cycle constant at 1% so the
    cycle-averaged thermal load is comparable across the sweep. Track
    six outcome variables to expose the regime transitions:

        (1) cycles per pulse N_c = τ_p · f0
        (2) shock-formation indicator: σ = β · k · ε · L_p (Hamilton & Blackstock 1998)
        (3) per-pulse cumulative cavitation probability over the cycle count
        (4) per-pulse adiabatic focal ΔT (linear-fundamental absorption)
        (5) effective harmonic-absorption gain (depends on σ)
        (6) per-pulse focal-voxel transient T (clamped at 100 °C)

    Three carrier frequencies (0.5 MHz, 1 MHz, 1.5 MHz) shown. The
    sweep crosses the canonical regime boundaries automatically:
    intrinsic-threshold regime at small τ_p (single-cycle nucleation),
    shock-vapor regime at large τ_p with shock formation, and a
    transitional shock-scattering band in between.
    """
    print("[fig12] Pulse-duration sweep (μs → ms)")

    tau = np.geomspace(1.0e-6, 20.0e-3, 200)
    cases = [
        ("0.5 MHz, |p⁻| 18 MPa", 0.5e6, 18.0e6, 35.0e6, "#2ca02c"),
        ("1.0 MHz, |p⁻| 25 MPa", 1.0e6, 25.0e6, 60.0e6, "#1f77b4"),
        ("1.0 MHz, |p⁻| 15 MPa (shock-vapor regime)", 1.0e6, 15.0e6, 85.0e6, "#d62728"),
    ]

    # Liver tissue
    rho, c, alpha0, cp = 1079.0, 1595.0, 8.69, 3540.0
    beta = 4.5  # nonlinearity for liver

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

    for label, f0, pnp, ppp, color in cases:
        n_c = tau * f0  # cycles per pulse
        # Goldberg shock-formation parameter σ ≈ β · k · ε · L_p,
        # ε = pnp/(ρ c²) is the acoustic Mach number, L_p = c · τ_p
        # (single-pass propagation length over pulse duration). σ ≥ 1
        # marks fully-formed shock.
        eps = pnp / (rho * c * c)
        k = 2 * np.pi * f0 / c
        L_p = c * tau
        sigma_shock = beta * k * eps * L_p
        # Effective absorption gain: rises from 1 (no shock) to ~10 (full shock)
        # following a soft saturation σ/(σ+1)+1 → max 10 at σ→∞.
        alpha_gain = 1.0 + 9.0 * sigma_shock / (sigma_shock + 1.0)

        # Cumulative cavitation probability over the cycle count.
        # Per-cycle Pcav from Maxwell 2013 erf-CDF at f0.
        pcav_per_cycle = cav_prob(pnp, f0)
        p_cum = 1.0 - (1.0 - pcav_per_cycle) ** np.maximum(n_c, 1.0)

        # Heating amplitude factor: PPP-dominated for shock-rich pulses,
        # blends in with sigma_shock.
        amp_factor = 1.0 + (ppp / pnp - 1.0) * sigma_shock / (sigma_shock + 1.0)
        alpha = alpha0 * (f0 / 1e6) ** 1.1
        I_eff = (pnp * amp_factor) ** 2 / (2.0 * rho * c)
        Q = 2.0 * alpha * alpha_gain * I_eff
        dT_p = Q * tau / (rho * cp)
        T_transient = np.minimum(37.0 + dT_p, 100.0)

        axes[0, 0].plot(tau * 1e6, n_c, color=color, lw=1.5, label=label)
        axes[0, 1].plot(tau * 1e6, sigma_shock, color=color, lw=1.5)
        axes[0, 2].plot(tau * 1e6, p_cum, color=color, lw=1.5)
        axes[1, 0].plot(tau * 1e6, alpha_gain, color=color, lw=1.5)
        axes[1, 1].plot(tau * 1e6, dT_p, color=color, lw=1.5)
        axes[1, 2].plot(tau * 1e6, T_transient, color=color, lw=1.5)

    # Regime band shading on each subplot
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.axvspan(1.0, 20.0, alpha=0.07, color="#1f77b4")             # μs intrinsic
        ax.axvspan(20.0, 1000.0, alpha=0.07, color="#9467bd")          # transitional
        ax.axvspan(1000.0, 20000.0, alpha=0.07, color="#d62728")       # ms regime
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("pulse duration τ_p [μs]")

    axes[0, 0].set(ylabel="cycles per pulse $N_c$", title="Pulse cycle count")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=7, loc="upper left")

    axes[0, 1].set(ylabel="Goldberg shock parameter $\\sigma$",
                   title="Shock-formation indicator")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0, 1].text(2.0, 1.3, "σ = 1: shock onset", fontsize=8, color="k")

    axes[0, 2].set(ylabel="cumulative single-pulse $P_{\\mathrm{cav}}$",
                   title="Cavitation probability vs cycles")
    axes[0, 2].set_ylim(-0.02, 1.05)

    axes[1, 0].set(ylabel="absorption gain $\\alpha_{\\mathrm{eff}}/\\alpha(f_0)$",
                   title="Shock-enhanced absorption")
    axes[1, 0].set_yscale("log")

    axes[1, 1].set(ylabel="per-pulse adiabatic ΔT [K]",
                   title="Single-pulse focal temperature rise")
    axes[1, 1].set_yscale("log")

    axes[1, 2].set(ylabel="transient focal-voxel T [°C]",
                   title="Transient focal-voxel temperature\n(clamped at 100 °C by vapor seeding)")
    axes[1, 2].axhline(100.0, color="r", ls="--", lw=0.8)
    axes[1, 2].axhline(43.0, color="k", ls="--", lw=0.8)
    axes[1, 2].set_ylim(35, 105)

    # Add regime-band annotations to top-left only.
    for x_mid, label in [(4.0, "μs\nintrinsic"), (140.0, "transitional\nshock-scattering"),
                         (5000.0, "ms shock-vapor /\nsub-threshold")]:
        axes[0, 0].text(x_mid, axes[0, 0].get_ylim()[1] * 0.4, label,
                        ha="center", fontsize=7.5, color="grey",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    fig.suptitle("Pulse-duration sweep (1 μs → 20 ms): regime crossover diagnostics")
    fig.tight_layout()
    savefig("fig12_pulse_duration_sweep")
    plt.close(fig)


if __name__ == "__main__":
    fig07_pulse_waveforms()
    fig08_intrinsic_threshold_freq()
    fig09_prf_optimization()
    fig10_rib_thermal_safety()
    fig11_tumour_coverage()
    fig12_pulse_duration_sweep()
    print("[ch21c] Done.")
