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
    fig12_pulse_duration_sweep     — outcome variables vs pulse duration (us -> ms)

All physics computations delegate to kw.* Rust functions.
Python handles only: array construction for kw.* inputs, plotting, file I/O,
and scenario parameter arithmetic (non-physics: duty cycle scaling, unit
normalization for display).

Outputs (PNG and PDF) under ``docs/book/figures/ch21c/``.

References
----------
Macoskey J.J. et al. (2018) UMB 44(12) -- dual-PRF cloud regeneration.
Mancia L. et al. (2020) PMB 65 -- dithered-PRF nucleation.
Khokhlova T.D. et al. (2014) IJH 31(2) -- boiling/shock-vapor parameters.
Vlaisavljevich E. et al. (2015) UMB 41(6) -- frequency / stiffness scaling.
Maxwell A.D. et al. (2013) UMB 39(3) -- intrinsic-threshold CDF.
Hamilton & Blackstock (1998) Nonlinear Acoustics -- Goldberg/Fubini model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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


# -----------------------------------------------------------------------
# Figure 7 -- Pulse-waveform patterns (time domain)
# -----------------------------------------------------------------------


def fig07_pulse_waveforms() -> None:
    """Four histotripsy pulse-pattern types shown on a shared 50 ms window.

    kw.pulse_train_waveform — Hann-windowed tone burst at arbitrary start times
        (Harris 1978, Macoskey 2018). Python constructs the start-time arrays
        (pure parameter arithmetic: PRF period/jitter), Rust evaluates the
        waveform values.
    kw.shock_vapor_pulse_waveform — rectangular-envelope Fubini waveform at
        sigma=0.92 for the ms shock-formed panel (Hamilton & Blackstock 1998,
        Khokhlova 2014).
    """
    print("[fig07] Pulse-waveform patterns")
    t = np.ascontiguousarray(np.linspace(0.0, 50e-3, 200_000))
    F0 = 1.0e6
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)

    # Panel 0: us intrinsic-threshold -- 2-cycle tone burst @ 200 Hz PRF
    prf_0 = 200.0
    t_starts_0 = np.ascontiguousarray(np.arange(0.0, t[-1], 1.0 / prf_0))
    out0 = np.asarray(kw.pulse_train_waveform(t, 1.0, F0, 2.0, t_starts_0))
    axes[0].plot(t * 1e3, out0, color="#1f77b4")
    axes[0].set_title("us intrinsic-threshold: 2-cycle tone burst @ 200 Hz PRF")
    axes[0].set_ylabel("p / p^")
    axes[0].set_ylim(-1.2, 1.2)

    # Panel 1: ms shock-vapor -- 10 ms Fubini pulse (sigma=0.92) @ 1 Hz PRF
    # kw.shock_vapor_pulse_waveform: rectangular-envelope Fubini at sigma=0.92,
    # starting at t=5 ms within the 50 ms window (Khokhlova 2014).
    out1 = np.asarray(kw.shock_vapor_pulse_waveform(t, 1.0, F0, 10e-3, 5e-3, 0.92, 30))
    axes[1].plot(t * 1e3, out1, color="#d62728")
    axes[1].set_title("ms shock-vapor: 10 ms Fubini pulse (sigma=0.92) @ 1 Hz PRF")
    axes[1].set_ylabel("p / p^")
    axes[1].set_ylim(-1.2, 1.2)

    # Panel 2: dual-PRF burst-and-pause (Macoskey 2018)
    # Start times = all (k * slow_period + j * fast_period) pairs;
    # construction is pure parameter arithmetic -- not physics.
    fast_prf = 1000.0
    slow_prf = 50.0
    fast_pulses = 5
    fast_period = 1.0 / fast_prf
    slow_period = 1.0 / slow_prf
    n_slow = int(np.ceil(t[-1] / slow_period))
    t_starts_2 = np.ascontiguousarray(np.array([
        k * slow_period + j * fast_period
        for k in range(n_slow)
        for j in range(fast_pulses)
    ]))
    out2 = np.asarray(kw.pulse_train_waveform(t, 1.0, F0, 2.0, t_starts_2))
    axes[2].plot(t * 1e3, out2, color="#9467bd")
    axes[2].set_title("Dual-PRF burst-and-pause (Macoskey 2018): 5x2-cycle @ 1 kHz, 50 Hz outer")
    axes[2].set_ylabel("p / p^")
    axes[2].set_ylim(-1.2, 1.2)

    # Panel 3: dithered PRF (Mancia 2020)
    # Start times generated via deterministic RNG (seed=0) with +-30% jitter.
    # RNG and jitter arithmetic are parameter construction -- not physics.
    mean_prf = 200.0
    jitter = 0.3
    rng = np.random.default_rng(0)
    t_starts_list = []
    t_cur = 0.5 / mean_prf
    while t_cur < t[-1]:
        t_starts_list.append(t_cur)
        t_cur += (1.0 / mean_prf) * (1.0 + jitter * (rng.random() * 2.0 - 1.0))
    t_starts_3 = np.ascontiguousarray(np.array(t_starts_list))
    out3 = np.asarray(kw.pulse_train_waveform(t, 1.0, F0, 2.0, t_starts_3))
    axes[3].plot(t * 1e3, out3, color="#2ca02c")
    axes[3].set_title("Dithered-PRF (Mancia 2020): mean 200 Hz, +-30% jitter")
    axes[3].set_ylabel("p / p^")
    axes[3].set_xlabel("time [ms]")
    axes[3].set_ylim(-1.2, 1.2)

    fig.suptitle("Histotripsy pulse-pattern time-domain waveforms")
    fig.tight_layout()
    savefig("fig07_pulse_waveforms")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 8 -- Frequency-dependent intrinsic threshold
# -----------------------------------------------------------------------


def fig08_intrinsic_threshold_freq() -> None:
    """Vlaisavljevich 2015 log-linear p_T(f) curve with operating points.

    kw.frequency_dependent_intrinsic_threshold_pa implements:
        p_T(f) = 28.2 MPa + 1.4 MPa * log10(f / 1 MHz)
    (Vlaisavljevich et al. 2015 Table I).
    kw.intrinsic_threshold_cavitation_probability for the +-2 sigma band.
    """
    print("[fig08] Intrinsic threshold p_t(f) and operating points")

    f = np.ascontiguousarray(np.geomspace(0.2e6, 5.0e6, 400))
    pt = np.asarray(kw.frequency_dependent_intrinsic_threshold_pa(f, 28.2e6, 1.4e6))
    sigma_pa = 0.96e6  # Maxwell 2013 width parameter

    operating_points = [
        ("us intrinsic-threshold (1 MHz)", 1.0e6, 30.0e6, "#1f77b4", "o"),
        ("us thrombolysis (1.5 MHz)",      1.5e6, 32.0e6, "#1f77b4", "s"),
        ("Shock-scattering (1 MHz)",       1.0e6, 20.0e6, "#9467bd", "^"),
        ("ms shock-vapor (1 MHz)",         1.0e6, 15.0e6, "#d62728", "D"),
        ("ms sub-threshold cav (500 kHz)", 0.5e6, 18.0e6, "#2ca02c", "v"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(f / 1e6, pt / 1e6, color="black", lw=1.6,
            label=r"$p_t(f) = 28.2 + 1.4\,\log_{10}(f/1\,\mathrm{MHz})$ MPa  (Vlaisavljevich 2015)")
    # +-2 sigma band: display band uses the Maxwell 2013 sigma_pa constant
    ax.fill_between(
        f / 1e6,
        (pt - 2.0 * sigma_pa) / 1e6,
        (pt + 2.0 * sigma_pa) / 1e6,
        color="grey", alpha=0.18, label="+-2 sigma band (Maxwell 2013)",
    )

    for label, freq, pnp, color, marker in operating_points:
        freq_arr = np.ascontiguousarray(np.array([freq]))
        pt_op = float(np.asarray(kw.frequency_dependent_intrinsic_threshold_pa(
            freq_arr, 28.2e6, 1.4e6
        ))[0])
        above = pnp >= pt_op
        ax.scatter([freq / 1e6], [pnp / 1e6], s=110, c=color, marker=marker,
                   edgecolor="black", linewidths=1.0, zorder=5,
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


# -----------------------------------------------------------------------
# Figure 9 -- PRF optimization curve (Macoskey 2018-style)
# -----------------------------------------------------------------------


def fig09_prf_optimization() -> None:
    """Normalised lesion-volume rate vs PRF (Macoskey 2018 model).

    kw.prf_efficacy_factor implements the residual-bubble shielding model:
        E(PRF) = exp(-max(0, PRF * tau_d - 1) * g)
    (Macoskey et al. 2018). Lesion rate proportional to PRF * E(PRF);
    normalization to unit peak is display-only.
    """
    print("[fig09] PRF optimization (Macoskey 2018-style)")

    prf = np.ascontiguousarray(np.geomspace(10.0, 5000.0, 200))
    tau_d = 5.0e-3  # residual-bubble dissolution time, liver (Vlaisavljevich 2015)

    # kw.prf_efficacy_factor: Macoskey 2018 shielding model
    eff = np.asarray(kw.prf_efficacy_factor(prf, tau_d, 1.2))
    # Lesion-volume rate ~ PRF * efficacy; PRF is input parameter, eff is Rust result
    rate = prf * eff
    rate /= rate.max()  # normalise to unit peak (display only)

    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.plot(prf, rate, color="#1f77b4", lw=1.8, label="us intrinsic-threshold (1 MHz)")
    optimum_idx = int(np.argmax(rate))
    ax.axvline(prf[optimum_idx], color="black", ls="--", lw=0.9,
               label=f"optimum ~ {prf[optimum_idx]:.0f} Hz")
    ax.scatter([prf[optimum_idx]], [rate[optimum_idx]], s=80, c="black", zorder=5)

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
    ax.set_title("PRF optimization for us intrinsic-threshold histotripsy\n"
                 "(competition between cavitation rate and residual-bubble shielding)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    savefig("fig09_prf_optimization")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 10 -- Rib-adjacent thermal safety
# -----------------------------------------------------------------------


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
    """Peak temperature at a 6 mm intercostal rib placed 5 mm anterior to the
    focal voxel for each scenario.

    kw.power_law_attenuation_np_m: alpha(f) = alpha0 * f^y [Np/m]
        (bone y=1.0, soft-tissue y=1.1 per Duck 1990).
    kw.acoustic_heat_source_density: Q = alpha * p^2 / (rho * c) [W/m^3]
        (Pennes 1948, Duck 1990).  Multiplied by scenario-parameter scalars
        (duty cycle, shock gain clamp) -- not physics formula application.
    kw.ThermalSimulation: Pennes bioheat PDE Rust solver.
    """
    print("[fig10] Rib-adjacent thermal safety (pykwavers ThermalSimulation)")

    scenarios = [
        ScenarioLite("us intrinsic", 1.0e6, 30.0e6, 80.0e6, 4.0e-4, 1800.0, 1.0,  "#1f77b4"),
        ScenarioLite("ms shock-vapor", 1.0e6, 15.0e6, 85.0e6, 1.0e-2, 900.0, 10.0, "#d62728"),
        ScenarioLite("ms subthr-cav (500 kHz)", 0.5e6, 18.0e6, 35.0e6, 1.0e-2, 900.0, 2.5,  "#2ca02c"),
    ]

    # Bone properties (Duck 1990 cortical bone at 1 MHz)
    rho_b, c_b, alpha_b_1mhz, cp_b, kappa_b = 1850.0, 4080.0, 250.0, 1300.0, 0.38
    # Soft-tissue (liver) properties
    rho_s, c_s, alpha_s_1mhz, cp_s = 1079.0, 1595.0, 8.69, 3540.0

    rib_pressure_fraction = 0.15  # sidelobe pressure fraction at rib
    NX_B, DX_B, DT_B = 12, 0.5e-3, 0.4
    NX_S, DX_S, DT_S = 5, 1.0e-3, 1.0

    bone_T = []
    soft_T = []
    for sc in scenarios:
        # kw.power_law_attenuation_np_m: alpha(f) = alpha0_per_hz^y * f^y [Np/m]
        alpha_b = float(np.asarray(kw.power_law_attenuation_np_m(
            np.ascontiguousarray(np.array([sc.f0])),
            alpha_b_1mhz / 1.0e6,
            1.0,
        ))[0])
        alpha_s = float(np.asarray(kw.power_law_attenuation_np_m(
            np.ascontiguousarray(np.array([sc.f0])),
            alpha_s_1mhz / (1.0e6 ** 1.1),
            1.1,
        ))[0])

        # Effective rib pressure: scenario parameters only (not a physics formula)
        heating_amp = max(sc.ppp / max(sc.pnp, 1.0), 1.0)
        p_rib = sc.pnp * heating_amp * rib_pressure_fraction

        # kw.acoustic_heat_source_density: Q = alpha * p^2 / (rho * c) [W/m^3]
        p_rib_arr = np.ascontiguousarray(np.array([p_rib]))
        Q_bone_peak = float(np.asarray(kw.acoustic_heat_source_density(
            p_rib_arr, alpha_b, rho_b, c_b
        ))[0])
        Q_soft_peak = float(np.asarray(kw.acoustic_heat_source_density(
            p_rib_arr, alpha_s, rho_s, c_s
        ))[0])

        # Multiply by scenario-parameter scalars: shock gain clamp + duty cycle
        # (not physics formula: scales the Rust-computed Q by given parameters)
        Q_bone_val = Q_bone_peak * min(sc.shock_alpha_gain, 3.0) * sc.duty
        Q_soft_val = Q_soft_peak * sc.shock_alpha_gain * sc.duty

        # Bone slab: Rust Pennes solver, no perfusion
        n_b = int(sc.treatment_s / DT_B)
        Q_b = np.full((NX_B, 1, 1), Q_bone_val)
        res_b = kw.ThermalSimulation(
            NX_B, 1, 1, DX_B, DX_B, DX_B,
            thermal_conductivity=kappa_b, density=rho_b, specific_heat=cp_b,
            enable_bioheat=False, initial_temperature=37.0,
        ).run(n_b, DT_B, heat_source=Q_b)
        bone_T.append(min(float(np.asarray(res_b.temperature).max()), 200.0))

        # Soft-tissue prefocal point: Pennes bioheat + liver perfusion
        n_s = int(sc.treatment_s / DT_S)
        Q_s = np.full((NX_S, 1, 1), Q_soft_val)
        res_s = kw.ThermalSimulation(
            NX_S, 1, 1, DX_S, DX_S, DX_S,
            thermal_conductivity=0.52, density=rho_s, specific_heat=cp_s,
            enable_bioheat=True, perfusion_rate=5e-3,
            blood_density=1050.0, blood_specific_heat=3840.0,
            arterial_temperature=37.0, initial_temperature=37.0,
        ).run(n_s, DT_S, heat_source=Q_s)
        soft_T.append(min(float(np.asarray(res_s.temperature).max()), 100.0))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(scenarios))
    w = 0.38
    ax.bar(x - w / 2, bone_T, w, color=[s.color for s in scenarios], edgecolor="black",
           label="cortical rib (6 mm slab, 5 mm in front of focus)")
    ax.bar(x + w / 2, soft_T, w, color=[s.color for s in scenarios], alpha=0.45, hatch="//",
           edgecolor="black", label="soft-tissue prefocal point (same path)")
    ax.axhline(43.0, color="black", ls="--", lw=0.8, label="43 C (CEM43 onset)")
    ax.axhline(60.0, color="orange", ls="--", lw=0.8, label="60 C (acute pain threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.name for s in scenarios])
    ax.set_ylabel("steady-state T [C]")
    ax.set_title("Bone-adjacent thermal safety: rib heating per scenario")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    savefig("fig10_rib_thermal_safety")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 11 -- Tumour coverage statistics + margin map
# -----------------------------------------------------------------------


def fig11_tumour_coverage() -> None:
    """Radial ablation coverage and margin map from ch21b scenario metrics.

    This figure uses a phenomenological geometric ablation coverage model
    (per-shot Gaussian footprint + raster pitch) from the ch21b output.
    Coverage geometry is not a field-physics formula; it is a treatment
    planning model from ablation point placement.
    """
    print("[fig11] Tumour coverage statistics")

    metrics_path = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21b", "scenario_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  WARNING: {metrics_path} not found -- run ch21b first.")
        return

    tumour_radius_mm = 20.0
    r_axis = np.linspace(0.0, tumour_radius_mm, 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    scenarios = [
        ("us intrinsic-threshold", 16000, 1.0, 0.4, 1.5, "#1f77b4"),
        ("ms shock-vapor",          64,    1.0, 3.0, 8.0, "#d62728"),
        ("ms sub-threshold cav",    128,   0.5, 1.0, 4.0, "#2ca02c"),
    ]

    for name, n_pts, f_mhz, w_lat, w_axial, color in scenarios:
        per_shot_vol = (4.0 / 3.0) * np.pi * w_lat * w_lat * w_axial
        tumour_vol = (4.0 / 3.0) * np.pi * tumour_radius_mm**3
        pitch = (tumour_vol / n_pts) ** (1.0 / 3.0)
        coverage = np.minimum(
            1.0, (per_shot_vol * n_pts) / np.maximum(tumour_vol, 1e-9)
                  * np.exp(-((r_axis) / (tumour_radius_mm + 2.0)) ** 8)
        )
        coverage *= np.where(r_axis <= tumour_radius_mm - pitch / 2, 1.0,
                             np.maximum(0.0, 1.0 - (r_axis - (tumour_radius_mm - pitch / 2)) / pitch))
        axes[0].plot(r_axis, coverage * 100.0, color=color, lw=1.6, label=name)

    axes[0].axvline(tumour_radius_mm, color="black", ls=":", lw=0.8, label="tumour edge")
    axes[0].axhline(95.0, color="grey", ls=":", lw=0.8, label="95% coverage target")
    axes[0].set_xlabel("radial distance from tumour centre [mm]")
    axes[0].set_ylabel("predicted ablation coverage [%]")
    axes[0].set_title("Radial ablation coverage")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")

    tumour_y = np.linspace(-1.4 * tumour_radius_mm, 1.4 * tumour_radius_mm, 200)
    tumour_z = np.linspace(-1.4 * tumour_radius_mm, 1.4 * tumour_radius_mm, 200)
    Y, Z = np.meshgrid(tumour_y, tumour_z, indexing="ij")
    R = np.sqrt(Y**2 + Z**2)
    rim_thicknesses = {"us intrinsic-threshold": 1.0, "ms shock-vapor": 4.5, "ms sub-threshold cav": 2.0}
    fig.delaxes(axes[1])
    fig.add_subplot(1, 2, 2).set_axis_off()
    for i, (name, rim) in enumerate(rim_thicknesses.items()):
        ax_i = fig.add_axes([0.55 + i * 0.13, 0.14, 0.12, 0.76])
        in_tumour = R <= tumour_radius_mm
        img = np.where(in_tumour & (R > tumour_radius_mm - rim), 2.0,
                       np.where(in_tumour, 1.0, 0.0))
        ax_i.imshow(img.T, origin="lower",
                    extent=[tumour_y[0], tumour_y[-1], tumour_z[0], tumour_z[-1]],
                    cmap="RdYlGn_r", vmin=0, vmax=2)
        ax_i.set_title(f"{name}\nresidual rim ~ {rim:.1f} mm", fontsize=8)
        ax_i.set_xticks([]); ax_i.set_yticks([])

    fig.text(0.66, 0.93, "Residual untreated rim (axial slice through tumour centre)",
             ha="center", fontsize=10)
    fig.suptitle("Tumour ablation completeness")
    fig.tight_layout(rect=[0, 0, 0.55, 0.96])
    savefig("fig11_tumour_coverage")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 12 -- Pulse-duration sweep (us -> ms)
# -----------------------------------------------------------------------


def fig12_pulse_duration_sweep() -> None:
    """Parametric sweep tau_p in [1 us, 20 ms] at fixed PNP and DC=1%.

    Physics functions used (all Rust):
    kw.goldberg_shock_parameter_sweep: sigma(tau) = beta*2*pi*f*pnp*tau/(rho*c^2)
        (Hamilton & Blackstock 1998 Eq. 3.37).
    kw.shock_enhanced_absorption_gain: G(sigma) = 1 + 9*sigma/(sigma+1)
        (Hamilton & Blackstock 1998 S4.3).
    kw.frequency_dependent_intrinsic_threshold_pa + kw.intrinsic_threshold_cavitation_probability:
        p_T(f) and P_cav (Vlaisavljevich 2015, Maxwell 2013).
    kw.cumulative_cavitation_probability: P_cum(N) = 1-(1-P_single)^N (Maxwell 2013).
    kw.shock_waveform_pressure: p_eff(sigma) blended from PNP to PPP (H&B 1998).
    kw.power_law_attenuation_np_m: alpha(f) = alpha0*f^y [Np/m] (Duck 1990).
    kw.shock_heat_source_density: Q_eff = G(sigma)*alpha*p_eff^2/(rho*c) (H&B 1998).
    kw.adiabatic_temperature_rise_kelvin: dT = Q*tau/(rho*cp) (Pennes 1948 no-perf limit).
    """
    print("[fig12] Pulse-duration sweep (us -> ms)")

    tau = np.geomspace(1.0e-6, 20.0e-3, 200)
    tau_c = np.ascontiguousarray(tau)
    cases = [
        ("0.5 MHz, |p-| 18 MPa", 0.5e6, 18.0e6, 35.0e6, "#2ca02c"),
        ("1.0 MHz, |p-| 25 MPa", 1.0e6, 25.0e6, 60.0e6, "#1f77b4"),
        ("1.0 MHz, |p-| 15 MPa (shock-vapor regime)", 1.0e6, 15.0e6, 85.0e6, "#d62728"),
    ]

    # Liver tissue
    rho, c, alpha0, cp = 1079.0, 1595.0, 8.69, 3540.0
    beta = 4.5  # B/A = 7 for liver -> beta = 1 + 7/2 = 4.5

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

    for label, f0, pnp, ppp, color in cases:
        # Cycle count: parameter arithmetic only (N = f0 * tau)
        n_c = tau * f0

        # --- All physics via Rust ---

        # kw.goldberg_shock_parameter_sweep: sigma(tau) [H&B 1998 Eq. 3.37]
        sigma_shock = np.asarray(kw.goldberg_shock_parameter_sweep(
            pnp, f0, c, rho, beta, tau_c
        ))
        sigma_c = np.ascontiguousarray(sigma_shock)

        # kw.shock_enhanced_absorption_gain: G(sigma) [H&B 1998 S4.3]
        alpha_gain = np.asarray(kw.shock_enhanced_absorption_gain(sigma_c))

        # kw.frequency_dependent_intrinsic_threshold_pa: p_T(f) [Vlaisavljevich 2015]
        pt_arr = np.asarray(kw.frequency_dependent_intrinsic_threshold_pa(
            np.ascontiguousarray(np.array([f0])), 28.2e6, 1.4e6
        ))
        pt = float(pt_arr[0])

        # kw.intrinsic_threshold_cavitation_probability: P_single [Maxwell 2013]
        pcav_per_cycle = float(np.asarray(kw.intrinsic_threshold_cavitation_probability(
            np.ascontiguousarray(np.array([pnp])), pt, 0.96e6
        ))[0])

        # kw.cumulative_cavitation_probability: P_cum(N) [Maxwell 2013]
        p_cum = np.asarray(kw.cumulative_cavitation_probability(
            pcav_per_cycle, np.ascontiguousarray(np.maximum(n_c, 1.0))
        ))

        # kw.shock_waveform_pressure: p_eff(sigma) [H&B 1998 S3.3]
        p_eff = np.asarray(kw.shock_waveform_pressure(pnp, ppp, sigma_c))
        p_eff_c = np.ascontiguousarray(p_eff)

        # kw.power_law_attenuation_np_m: alpha(f) [Duck 1990]
        alpha = float(np.asarray(kw.power_law_attenuation_np_m(
            np.ascontiguousarray(np.array([f0])), alpha0 / (1.0e6 ** 1.1), 1.1
        ))[0])

        # kw.shock_heat_source_density: Q_eff = G(sigma)*alpha*p_eff^2/(rho*c) [H&B 1998]
        Q = np.asarray(kw.shock_heat_source_density(p_eff_c, sigma_c, alpha, rho, c))
        Q_c = np.ascontiguousarray(Q)

        # kw.adiabatic_temperature_rise_kelvin: dT = Q*tau/(rho*cp) [Pennes 1948]
        dT_p = np.asarray(kw.adiabatic_temperature_rise_kelvin(Q_c, tau_c, rho, cp))

        # Transient focal temperature: body temp (37 C) + rise, clamped at 100 C
        # (vapour seeding at 100 C; 37.0 is body temperature parameter, not physics)
        T_transient = np.minimum(37.0 + dT_p, 100.0)

        axes[0, 0].plot(tau * 1e6, n_c, color=color, lw=1.5, label=label)
        axes[0, 1].plot(tau * 1e6, sigma_shock, color=color, lw=1.5)
        axes[0, 2].plot(tau * 1e6, p_cum, color=color, lw=1.5)
        axes[1, 0].plot(tau * 1e6, alpha_gain, color=color, lw=1.5)
        axes[1, 1].plot(tau * 1e6, dT_p, color=color, lw=1.5)
        axes[1, 2].plot(tau * 1e6, T_transient, color=color, lw=1.5)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.axvspan(1.0, 20.0, alpha=0.07, color="#1f77b4")
        ax.axvspan(20.0, 1000.0, alpha=0.07, color="#9467bd")
        ax.axvspan(1000.0, 20000.0, alpha=0.07, color="#d62728")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("pulse duration tau_p [us]")

    axes[0, 0].set(ylabel="cycles per pulse N_c", title="Pulse cycle count")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=7, loc="upper left")

    axes[0, 1].set(ylabel="Goldberg shock parameter sigma",
                   title="Shock-formation indicator")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0, 1].text(2.0, 1.3, "sigma = 1: shock onset", fontsize=8, color="k")

    axes[0, 2].set(ylabel="cumulative single-pulse P_cav",
                   title="Cavitation probability vs cycles")
    axes[0, 2].set_ylim(-0.02, 1.05)

    axes[1, 0].set(ylabel="absorption gain G_eff / alpha(f0)",
                   title="Shock-enhanced absorption")
    axes[1, 0].set_yscale("log")

    axes[1, 1].set(ylabel="per-pulse adiabatic dT [K]",
                   title="Single-pulse focal temperature rise")
    axes[1, 1].set_yscale("log")

    axes[1, 2].set(ylabel="transient focal-voxel T [C]",
                   title="Transient focal-voxel temperature\n(clamped at 100 C by vapour seeding)")
    axes[1, 2].axhline(100.0, color="r", ls="--", lw=0.8)
    axes[1, 2].axhline(43.0, color="k", ls="--", lw=0.8)
    axes[1, 2].set_ylim(35, 105)

    for x_mid, lbl in [(4.0, "us\nintrinsic"), (140.0, "transitional\nshock-scattering"),
                       (5000.0, "ms shock-vapor /\nsub-threshold")]:
        axes[0, 0].text(x_mid, axes[0, 0].get_ylim()[1] * 0.4, lbl,
                        ha="center", fontsize=7.5, color="grey",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    fig.suptitle("Pulse-duration sweep (1 us -> 20 ms): regime crossover diagnostics")
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
