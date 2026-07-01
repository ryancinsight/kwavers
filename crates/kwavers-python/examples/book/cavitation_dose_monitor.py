"""Shared orchestration for passive-cavitation harmonic-dose monitoring.

Used by ch24 (BBB LIFU) and ch21e (liver histotripsy). This module is pure
glue: every physics step is a call into the kwavers Rust core
(`volume_emission_spectrum`, `volume_emission_sweep`,
`simulate_population_emission`, `population_emission_sweep`,
`simulated_population_monitor_timeseries`, `cumulative_cavitation_dose`,
`cavitation_controller_pressure`). No domain math lives here; Python adapts
arguments, formats labels, and plots returned arrays.

Pipeline (InsighTec Exablate-style cavitation dose):
  drive each microbubble in the focal volume V_s (Keller–Miksis)
  → far-field acoustic emission p_sc(t) each receiver detects
  → power spectrum
  → incoherent power-sum across the bubble population = array/receiver
    integration over V_s
  → decompose into harmonic / subharmonic / ultraharmonic / broadband
  → stable cavitation dose = ∫(sub+ultra), inertial cavitation dose = ∫broadband.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BubbleMedium:
    """Liquid + bubble parameters for the Keller–Miksis driver."""
    rho: float          # liquid density [kg/m³]
    sigma: float        # (shell) surface tension [N/m]
    gamma: float        # polytropic index
    mu: float           # liquid dynamic viscosity [Pa·s]
    pv: float           # vapour pressure [Pa]
    c_l: float          # liquid sound speed [m/s]
    p0: float = 101_325.0  # ambient pressure [Pa]
    xi_s: float = 0.0   # shell viscosity [Pa·s·m] (0 = bare bubble)


def vs_emission_spectrum(
    kw,
    *,
    drive_pa: float,
    f0: float,
    r0_population_m,
    medium: BubbleMedium,
    n_cycles: float = 12.0,
    steps_per_cycle: int = 4000,
    r_obs_m: float = 5.0e-2,
    n_fft: int = 2048,
    transient_fraction: float = 0.4,
):
    """Receiver/array-integrated emission spectrum for one focal volume V_s.

    Drives every bubble radius in ``r0_population_m`` at ``drive_pa`` via the
    Rust Keller–Miksis solver, computes each one's far-field emission, and
    incoherently power-sums the per-bubble spectra (the V_s array integral).

    Returns ``(freqs, psd_vs)`` for the volume-integrated spectrum.
    """
    freqs, psd, _n_active = kw.volume_emission_spectrum(
        float(drive_pa),
        float(f0),
        np.atleast_1d(r0_population_m).astype(float),
        float(n_cycles),
        int(steps_per_cycle),
        float(r_obs_m),
        int(n_fft),
        float(transient_fraction),
        medium.p0,
        medium.rho,
        medium.sigma,
        medium.gamma,
        medium.mu,
        medium.pv,
        medium.c_l,
        medium.xi_s,
    )
    return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)


def simulate_population_emission(
    kw,
    *,
    drive_pa: float,
    f0: float,
    medium: BubbleMedium,
    n_bubbles: int,
    seed: int,
    r0_median_m: float = 1.5e-6,
    r0_sigma_ln: float = 0.4,
    n_cycles: float = 12.0,
    n_out: int = 8192,
    r_obs_m: float = 5.0e-2,
    n_fft: int = 4096,
    rel_halfwidth: float = 0.12,
    noise_floor: float = 0.0,
    thermal_effects: bool = False,
    coated: bool = False,
    chi: float = 0.5,
    shell_viscosity: float = 0.5,
    shell_thickness: float = 3.0e-9,
    sigma_initial: float = 0.04,
    steps_per_cycle: int = 2000,
):
    """Simulate the acoustic emission of a driven microbubble *population* with
    the TRUE production bubble-dynamics solvers.

    With ``coated=False`` each bubble is a free gas bubble driven by the adaptive
    Keller–Miksis solver (``simulate_bubble_emission``); with ``coated=True`` it
    is an encapsulated microbubble driven by the Marmottant shell model
    (``simulate_coated_bubble_emission``), whose buckling/rupture emits the
    clinical subharmonic at low drive. Either way the harmonic / subharmonic /
    ultraharmonic / broadband content is EMERGENT from the simulated nonlinear
    dynamics — nothing is tuned to a target shape.

    Each of ``n_bubbles`` bubbles has a log-normally distributed equilibrium
    radius and is driven at the real ``drive_pa`` by ``simulate_bubble_emission``
    (adaptive sub-stepping that survives inertial collapse — no fixed-step
    blow-up, so the real amplitude regime is reachable). Each bubble nucleates at
    a random delay and contributes with a random amplitude weight; the per-bubble
    emissions are coherently superposed in the time domain
    (``ensemble_emission_superposition``) and the Hann-windowed PSD is decomposed
    into bands. The harmonic / subharmonic / ultraharmonic / broadband content is
    therefore an EMERGENT property of the simulated nonlinear dynamics — nothing
    is tuned to a target shape.

    ``seed`` is passed directly to the Rust population simulator for
    reproducible figures.
    Returns dict: ``freqs``, ``psd``, ``fundamental``, ``subharmonic``,
    ``ultraharmonic``, ``broadband``, ``stable`` (sub+ultra), ``total``,
    ``n_active``, ``max_compression``, ``max_mach``.
    """
    (
        freqs,
        psd,
        fund,
        sub,
        ultra,
        broad,
        stable,
        total,
        n_active,
        max_c,
        max_m,
    ) = kw.simulate_population_emission(
        drive_pa,
        f0,
        int(n_bubbles),
        int(seed),
        r0_median_m,
        r0_sigma_ln,
        n_cycles,
        int(n_out),
        r_obs_m,
        rel_halfwidth,
        noise_floor,
        thermal_effects,
        coated,
        chi,
        shell_viscosity,
        shell_thickness,
        sigma_initial,
        int(steps_per_cycle),
        medium.p0,
        medium.rho,
        medium.c_l,
        medium.mu,
        medium.sigma,
        medium.pv,
        medium.gamma,
    )
    f = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    return {
        "freqs": f, "psd": psd,
        "fundamental": fund, "subharmonic": sub, "ultraharmonic": ultra,
        "broadband": broad, "stable": stable,
        "total": total, "n_active": int(n_active),
        "max_compression": max_c, "max_mach": max_m,
    }


def population_dose_vs_pressure(
    kw,
    *,
    pressures_pa,
    f0: float,
    medium: BubbleMedium,
    n_bubbles: int,
    seed: int = 0,
    **sim_kwargs,
):
    """Population-simulated stable / inertial / harmonic emission vs pressure.

    Like :func:`dose_vs_pressure` but each point is a full population emission
    simulation (genuine broadband), not a single-bubble line spectrum. Returns
    arrays keyed ``harmonic``, ``subharmonic``, ``ultraharmonic``, ``stable``,
    ``inertial`` (= broadband), ``signal`` (= stable + inertial).
    """
    pressures_pa = np.atleast_1d(pressures_pa).astype(float)
    sim_kwargs = dict(sim_kwargs)
    sim_kwargs.pop("medium", None)
    sim_kwargs.pop("n_bubbles", None)
    r0_median_m = sim_kwargs.pop("r0_median_m", 1.5e-6)
    r0_sigma_ln = sim_kwargs.pop("r0_sigma_ln", 0.4)
    n_cycles = sim_kwargs.pop("n_cycles", 12.0)
    n_out = sim_kwargs.pop("n_out", 8192)
    r_obs_m = sim_kwargs.pop("r_obs_m", 5.0e-2)
    rel_halfwidth = sim_kwargs.pop("rel_halfwidth", 0.12)
    noise_floor = sim_kwargs.pop("noise_floor", 0.0)
    thermal_effects = sim_kwargs.pop("thermal_effects", False)
    coated = sim_kwargs.pop("coated", False)
    chi = sim_kwargs.pop("chi", 0.5)
    shell_viscosity = sim_kwargs.pop("shell_viscosity", 0.5)
    shell_thickness = sim_kwargs.pop("shell_thickness", 3.0e-9)
    sigma_initial = sim_kwargs.pop("sigma_initial", 0.04)
    steps_per_cycle = sim_kwargs.pop("steps_per_cycle", 2000)
    if sim_kwargs:
        unknown = ", ".join(sorted(sim_kwargs))
        raise TypeError(f"unknown population-sweep kwargs: {unknown}")

    (
        harmonic,
        subharmonic,
        ultraharmonic,
        stable,
        inertial,
        signal,
        _n_active,
        _max_compression,
        _max_mach,
    ) = kw.population_emission_sweep(
        pressures_pa,
        f0,
        int(n_bubbles),
        int(seed),
        r0_median_m,
        r0_sigma_ln,
        n_cycles,
        int(n_out),
        r_obs_m,
        rel_halfwidth,
        noise_floor,
        thermal_effects,
        coated,
        chi,
        shell_viscosity,
        shell_thickness,
        sigma_initial,
        int(steps_per_cycle),
        medium.p0,
        medium.rho,
        medium.c_l,
        medium.mu,
        medium.sigma,
        medium.pv,
        medium.gamma,
    )
    return {
        "harmonic": np.asarray(harmonic, dtype=float),
        "subharmonic": np.asarray(subharmonic, dtype=float),
        "ultraharmonic": np.asarray(ultraharmonic, dtype=float),
        "stable": np.asarray(stable, dtype=float),
        "inertial": np.asarray(inertial, dtype=float),
        "signal": np.asarray(signal, dtype=float),
    }


def dose_vs_pressure(
    kw,
    *,
    pressures_pa,
    f0: float,
    r0_population_m,
    medium: BubbleMedium,
    rel_halfwidth: float = 0.04,
    noise_floor: float = 0.0,
    **spectrum_kwargs,
):
    """Stable / inertial / harmonic emission power vs drive pressure.

    For each drive pressure, builds the V_s-integrated spectrum and decomposes
    it. Returns a dict of arrays (same length as ``pressures_pa``):
      ``stable`` (sub+ultra), ``inertial`` (broadband), ``harmonic``,
      ``subharmonic``, ``ultraharmonic``.
    """
    pressures_pa = np.atleast_1d(pressures_pa).astype(float)
    spectrum_kwargs = dict(spectrum_kwargs)
    n_cycles = spectrum_kwargs.pop("n_cycles", 12.0)
    steps_per_cycle = spectrum_kwargs.pop("steps_per_cycle", 4000)
    r_obs_m = spectrum_kwargs.pop("r_obs_m", 5.0e-2)
    n_fft = spectrum_kwargs.pop("n_fft", 2048)
    transient_fraction = spectrum_kwargs.pop("transient_fraction", 0.4)
    if spectrum_kwargs:
        unknown = ", ".join(sorted(spectrum_kwargs))
        raise TypeError(f"unknown V_s pressure-sweep kwargs: {unknown}")
    (
        harmonic,
        subharmonic,
        ultraharmonic,
        stable,
        inertial,
        _n_active,
    ) = kw.volume_emission_sweep(
        pressures_pa,
        float(f0),
        np.atleast_1d(r0_population_m).astype(float),
        float(rel_halfwidth),
        float(noise_floor),
        float(n_cycles),
        int(steps_per_cycle),
        float(r_obs_m),
        int(n_fft),
        float(transient_fraction),
        medium.p0,
        medium.rho,
        medium.sigma,
        medium.gamma,
        medium.mu,
        medium.pv,
        medium.c_l,
        medium.xi_s,
    )
    return {
        "harmonic": np.asarray(harmonic, dtype=float),
        "subharmonic": np.asarray(subharmonic, dtype=float),
        "ultraharmonic": np.asarray(ultraharmonic, dtype=float),
        "stable": np.asarray(stable, dtype=float),
        "inertial": np.asarray(inertial, dtype=float),
    }


def closed_loop_sonication(
    kw,
    *,
    pressures_pa,
    stable_power,
    inertial_power,
    n_bursts: int,
    burst_duration_s: float,
    p_start_pa: float,
    stable_target: float,
    inertial_limit: float,
    gain: float = 0.08,
):
    """Run the InsighTec-style closed-loop dose controller over a sonication.

    ``stable_power``/``inertial_power`` are the swept emission-power curves vs
    ``pressures_pa`` (from :func:`dose_vs_pressure`); per-burst emission at the
    current pressure is read off them by interpolation, the controller
    (Rust ``cavitation_controller_pressure``) steps the pressure, and the
    cumulative stable / inertial doses are integrated in Rust.

    Returns dict with per-burst ``pressure``, ``stable_emission``,
    ``inertial_emission`` and cumulative ``stable_dose`` / ``inertial_dose``.
    """
    pressures_pa = np.asarray(pressures_pa, dtype=float)
    stable_power = np.asarray(stable_power, dtype=float)
    inertial_power = np.asarray(inertial_power, dtype=float)
    p_hist, se_hist, ie_hist, stable_dose, inertial_dose = (
        kw.closed_loop_cavitation_sonication(
            pressures_pa,
            stable_power,
            inertial_power,
            int(n_bursts),
            float(burst_duration_s),
            float(p_start_pa),
            float(stable_target),
            float(inertial_limit),
            float(gain),
        )
    )

    return {
        "pressure": np.asarray(p_hist, dtype=float),
        "stable_emission": np.asarray(se_hist, dtype=float),
        "inertial_emission": np.asarray(ie_hist, dtype=float),
        "stable_dose": np.asarray(stable_dose, dtype=float),
        "inertial_dose": np.asarray(inertial_dose, dtype=float),
    }


def monitor_timeseries(
    kw,
    *,
    pressures_pa,
    cavitation_power,
    n_pulses: int,
    prf_hz: float,
    p_start_pa: float,
    target_signal: float,
    inertial_cap: float,
    gain: float = 0.06,
    jitter_sigma: float = 0.35,
    goal_fraction: float = 0.85,
    seed: int = 0,
):
    """Per-pulse real-time cavitation-monitor trace (clinical "Acoustic Controls"
    + "Cavitation Dose" graphs).

    Each pulse fires at the controller-set pressure; the *measured* cavitation
    signal is the deterministic above-baseline emission ``cavitation_power(p)``
    (interpolated from the swept curve) times a seeded log-normal nucleation
    jitter — the stochastic pulse-to-pulse spikiness real passive-cavitation
    detectors record. The controller (Rust ``cavitation_controller_pressure``)
    reacts to the measured signal, so applied power tracks cavitation; the
    cumulative dose (Rust ``cumulative_cavitation_dose``) climbs toward a
    prescribed goal.

    Returns dict: ``t`` [s], ``cavitation_signal``, ``power_pct``,
    ``cumulative_dose``, ``goal``.
    """
    pressures_pa = np.asarray(pressures_pa, dtype=float)
    cavitation_power = np.asarray(cavitation_power, dtype=float)
    t, sig, pwr, cumulative, goal = kw.cavitation_monitor_timeseries(
        pressures_pa,
        cavitation_power,
        int(n_pulses),
        float(prf_hz),
        float(p_start_pa),
        float(target_signal),
        float(inertial_cap),
        float(gain),
        float(jitter_sigma),
        float(goal_fraction),
        int(seed),
    )
    return {
        "t": np.asarray(t, dtype=float),
        "cavitation_signal": np.asarray(sig, dtype=float),
        "power_pct": np.asarray(pwr, dtype=float),
        "cumulative_dose": np.asarray(cumulative, dtype=float),
        "goal": float(goal),
    }


def simulated_monitor_timeseries(
    kw,
    *,
    f0: float,
    medium: BubbleMedium,
    n_bubbles: int,
    n_pulses: int,
    prf_hz: float,
    p_start_pa: float,
    p_lo_pa: float,
    p_hi_pa: float,
    target_signal: float,
    inertial_cap: float,
    gain: float = 0.06,
    goal_fraction: float = 0.85,
    seed: int = 0,
    sim_kwargs=None,
):
    """Fully-simulated real-time monitor trace: one bubble-population emission
    simulation *per pulse*.

    Unlike :func:`monitor_timeseries` (which jitters a precomputed curve), each
    pulse here re-draws and drives a fresh microbubble population
    (:func:`simulate_population_emission`); the measured stable / broadband
    emission is genuine simulated cavitation, so the pulse-to-pulse spikiness is
    finite-population stochasticity, not an imposed model. The controller
    (Rust ``cavitation_controller_pressure``) reacts to the simulated broadband
    (inertial) and stable emission; the cumulative dose climbs to the goal.

    Returns the same dict shape as :func:`monitor_timeseries` plus
    ``stable_signal`` and ``broadband_signal``.
    """
    sim_kwargs = dict(sim_kwargs or {})
    # `medium`/`n_bubbles` are passed explicitly below; drop any duplicates that
    # rode in via a shared population-config dict.
    sim_kwargs.pop("medium", None)
    sim_kwargs.pop("n_bubbles", None)
    r0_median_m = sim_kwargs.pop("r0_median_m", 1.5e-6)
    r0_sigma_ln = sim_kwargs.pop("r0_sigma_ln", 0.4)
    n_cycles = sim_kwargs.pop("n_cycles", 12.0)
    n_out = sim_kwargs.pop("n_out", 8192)
    r_obs_m = sim_kwargs.pop("r_obs_m", 5.0e-2)
    rel_halfwidth = sim_kwargs.pop("rel_halfwidth", 0.12)
    noise_floor = sim_kwargs.pop("noise_floor", 0.0)
    thermal_effects = sim_kwargs.pop("thermal_effects", False)
    coated = sim_kwargs.pop("coated", False)
    chi = sim_kwargs.pop("chi", 0.5)
    shell_viscosity = sim_kwargs.pop("shell_viscosity", 0.5)
    shell_thickness = sim_kwargs.pop("shell_thickness", 3.0e-9)
    sigma_initial = sim_kwargs.pop("sigma_initial", 0.04)
    steps_per_cycle = sim_kwargs.pop("steps_per_cycle", 2000)
    if sim_kwargs:
        unknown = ", ".join(sorted(sim_kwargs))
        raise TypeError(f"unknown simulated-monitor population kwargs: {unknown}")
    (
        t,
        sig,
        pwr,
        cumulative,
        goal,
        stable,
        broad,
    ) = kw.simulated_population_monitor_timeseries(
        f0,
        int(n_bubbles),
        int(n_pulses),
        prf_hz,
        p_start_pa,
        p_lo_pa,
        p_hi_pa,
        target_signal,
        inertial_cap,
        gain,
        goal_fraction,
        int(seed),
        r0_median_m,
        r0_sigma_ln,
        n_cycles,
        int(n_out),
        r_obs_m,
        rel_halfwidth,
        noise_floor,
        thermal_effects,
        coated,
        chi,
        shell_viscosity,
        shell_thickness,
        sigma_initial,
        int(steps_per_cycle),
        medium.p0,
        medium.rho,
        medium.c_l,
        medium.mu,
        medium.sigma,
        medium.pv,
        medium.gamma,
    )
    return {
        "t": np.asarray(t, dtype=float),
        "cavitation_signal": np.asarray(sig, dtype=float),
        "power_pct": np.asarray(pwr, dtype=float),
        "cumulative_dose": np.asarray(cumulative, dtype=float),
        "goal": float(goal),
        "stable_signal": np.asarray(stable, dtype=float),
        "broadband_signal": np.asarray(broad, dtype=float),
    }


def simulate_raster_pulsing(
    kw,
    *,
    spot_lateral_m,
    spot_axial_m,
    p_target_pa: float,
    f0_hz: float,
    c_m_s: float,
    cav_pressures_pa,
    cav_dose_per_pulse,
    pulses_per_spot: int,
    prf_hz: float,
    schedule: str,
    interleave_group: int = 0,
    attenuation_np_m: float = 0.0,
    apodized: bool = True,
    tau_dissolution_s: float = 5.0e-3,
    shielding_g: float = 1.2,
    tau_thermal_s: float = 1.0,
    thermal_gain_k_per_pulse: float = 0.0,
    goal_dose: float = 0.0,
    n_time_samples: int = 240,
):
    """Time-resolved subspot-grid pulsing under SEQUENTIAL vs INTERLEAVED order.

    The Rust ``raster_cavitation_pulsing`` core sets each subspot's delivered
    peak pressure from electronic-steering efficiency and tissue attenuation,
    then applies either firing order:
      * ``"sequential"`` — all ``pulses_per_spot`` pulses at spot 0, then spot 1,
        …; consecutive pulses at a spot are 1/PRF apart.
      * ``"interleaved"`` — round-robin over the ``interleave_group`` (default:
        all spots); consecutive pulses at a spot are ``group/PRF`` apart, so the
        spot rests while the others fire.

    Two physical memory effects make the orders differ — both emergent from the
    per-spot inter-pulse interval Δt:
      * residual-bubble shielding (Macoskey 2018), via the Rust steady
        per-spot efficacy at the effective per-spot PRF = 1/Δt — short Δt
        (sequential) shields and lowers per-pulse cavitation efficacy;
      * thermal accumulation — each pulse adds ΔT ∝ p²; between a spot's pulses
        the temperature relaxes by ``exp(−Δt/τ_thermal)``, so sequential builds
        heat at one spot while interleaved lets it cool.

    Returns dict: ``time`` [s], ``coverage`` (fraction of spots ≥ goal vs time),
    ``cumulative_dose`` (Σ effective dose vs time), ``per_spot_dose``,
    ``per_spot_peak_temp``, ``efficacy`` (steady per-pulse), ``dt_spot_s``,
    ``treatment_s``, ``lateral_mm``, ``axial_mm``.
    """
    lat = np.atleast_1d(spot_lateral_m).astype(float)
    ax = np.atleast_1d(spot_axial_m).astype(float)
    cav_pressures_pa = np.asarray(cav_pressures_pa, float)
    cav_dose_per_pulse = np.asarray(cav_dose_per_pulse, float)
    (
        time_s,
        coverage,
        cumulative_dose,
        per_spot_dose,
        per_spot_peak_temp,
        efficacy,
        dt_spot_s,
        treatment_s,
        p_spot_pa,
    ) = kw.raster_cavitation_pulsing(
        lat,
        ax,
        p_target_pa,
        f0_hz,
        c_m_s,
        cav_pressures_pa,
        cav_dose_per_pulse,
        pulses_per_spot,
        prf_hz,
        schedule,
        interleave_group,
        attenuation_np_m,
        apodized,
        tau_dissolution_s,
        shielding_g,
        tau_thermal_s,
        thermal_gain_k_per_pulse,
        goal_dose,
        n_time_samples,
    )

    return {
        "time": np.asarray(time_s, float),
        "coverage": np.asarray(coverage, float),
        "cumulative_dose": np.asarray(cumulative_dose, float),
        "per_spot_dose": np.asarray(per_spot_dose, float),
        "per_spot_peak_temp": np.asarray(per_spot_peak_temp, float),
        "efficacy": float(efficacy),
        "dt_spot_s": float(dt_spot_s),
        "treatment_s": treatment_s,
        "lateral_mm": lat * 1e3,
        "axial_mm": ax * 1e3,
        "p_spot_pa": np.asarray(p_spot_pa, float),
    }


def per_spot_dose(
    kw,
    *,
    lateral_offsets_m,
    axial_offsets_m,
    p_target_pa: float,
    f0_hz: float,
    c_m_s: float,
    pressures_pa,
    cavitation_power,
    n_pulses_per_spot: int,
    goal_pressure_pa: float,
    attenuation_np_m: float = 0.0,
    apodized: bool = True,
):
    """Per-subspot cumulative cavitation dose across the steered raster.

    For each (lateral, axial) steering offset from the mechanical focus the
    delivered peak pressure is derated by the electronic-steering efficiency
    (Rust ``electronic_steering_efficiency``) and by extra tissue attenuation
    along the axial offset (a stand-in for the depth-dependent
    attenuation/reflection/scattering that varies across the raster):
    ``p_spot = p_target · ε(Δlat, Δax) · exp(−α·max(Δax, 0))``.
    The per-spot dose is ``n_pulses · cavitation_power(p_spot)`` interpolated
    from the swept emission curve.

    Returns dict with 2-D arrays over the (axial × lateral) grid: ``dose``,
    ``efficiency``, ``p_spot`` plus the ``lateral_mm`` / ``axial_mm`` axes.
    """
    lat = np.asarray(lateral_offsets_m, dtype=float)
    ax = np.asarray(axial_offsets_m, dtype=float)
    pressures_pa = np.asarray(pressures_pa, dtype=float)
    cavitation_power = np.asarray(cavitation_power, dtype=float)

    dose, eff, p_spot, goal = kw.per_spot_cavitation_dose_grid(
        lat,
        ax,
        float(p_target_pa),
        float(f0_hz),
        float(c_m_s),
        pressures_pa,
        cavitation_power,
        int(n_pulses_per_spot),
        float(goal_pressure_pa),
        float(attenuation_np_m),
        bool(apodized),
    )
    grid_shape = (ax.size, lat.size)
    dose = np.asarray(dose, dtype=float).reshape(grid_shape)
    eff = np.asarray(eff, dtype=float).reshape(grid_shape)
    p_spot = np.asarray(p_spot, dtype=float).reshape(grid_shape)
    # Prescribed per-spot dose goal: the dose a spot receives at the minimum
    # effective drive pressure (defines the treatable steering envelope).
    return {
        "dose": dose,
        "efficiency": eff,
        "p_spot": p_spot,
        "lateral_mm": lat * 1e3,
        "axial_mm": ax * 1e3,
        "goal": goal,
    }


def plot_cavitation_monitor(
    *,
    spectrum,
    f0_hz,
    rel_halfwidth,
    timeseries,
    spot,
    title,
    modality_note="",
):
    """Render the 4-panel clinical-style real-time cavitation monitor.

    Panels mirror the Exablate sonication screen plus a spatial per-spot panel:
      (A) acoustic spectrum + cavitation-band filter windows (CFL → signal);
      (B) acoustic controls — cavitation signal + applied power vs time;
      (C) cavitation dose — cumulative toward the prescribed goal;
      (D) per-subspot cumulative dose across the steered raster.

    ``spectrum`` is ``(freqs_hz, psd)``; ``timeseries`` from
    :func:`monitor_timeseries`; ``spot`` from :func:`per_spot_dose`.
    Returns the matplotlib Figure (caller saves/closes).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    freqs, psd = np.asarray(spectrum[0]), np.asarray(spectrum[1])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title, fontsize=13, y=0.98)

    def _dark(ax):
        ax.set_facecolor("black")
        for s in ax.spines.values():
            s.set_color("#888")
        ax.tick_params(colors="#ccc", labelsize=8)
        ax.xaxis.label.set_color("#ddd")
        ax.yaxis.label.set_color("#ddd")

    # (A) Acoustic spectrum on a dB scale, with the clinical cavitation lines
    # marked: the subharmonic f0/2 is the stable-cavitation marker the receiver
    # tracks; the fundamental f0 is the drive; the (2n+1)f0/2 ultraharmonics are
    # the other stable markers; the inter-peak floor is the broadband (inertial)
    # signal. (No wide shaded bands — single lines, as on the device console.)
    from matplotlib.lines import Line2D
    axA = axes[0, 0]
    _dark(axA)
    f_norm = freqs / f0_hz
    psd_n = psd / (psd.max() + 1e-300)
    db = 10.0 * np.log10(psd_n + 1e-7)
    floor_db = -70.0
    axA.plot(f_norm, db, color="#ffae42", lw=0.8)
    axA.fill_between(f_norm, floor_db, db, color="#ffae42", alpha=0.30)
    axA.axvline(0.5, color="#39ff14", lw=1.8)                 # subharmonic f0/2
    axA.axvline(1.0, color="#4da6ff", lw=1.3)                 # fundamental f0
    for u in (1.5, 2.5):                                      # ultraharmonics
        axA.axvline(u, color="#39ff14", lw=0.9, ls=":")
    axA.axvline(2.0, color="#4da6ff", lw=0.8, ls=":")         # 2nd harmonic
    axA.set_xlim(0, 3.0)
    axA.set_ylim(floor_db, 5.0)
    axA.set_xlabel("Frequency / $f_0$")
    axA.set_ylabel("Emission PSD (dB re fundamental)")
    axA.set_title("(A) Acoustic spectrum — cavitation signatures\n"
                  "sub/ultraharmonics = stable-cavitation markers; floor = broadband (inertial)",
                  fontsize=9)
    handles = [
        Line2D([0], [0], color="#39ff14", lw=1.8, label="subharmonic $f_0/2$"),
        Line2D([0], [0], color="#4da6ff", lw=1.3, label="fundamental $f_0$"),
        Line2D([0], [0], color="#39ff14", lw=0.9, ls=":", label="ultraharmonics"),
    ]
    axA.legend(handles=handles, loc="upper right", fontsize=7, facecolor="black",
               edgecolor="#888", labelcolor="#ddd")

    # (B) Acoustic controls: cavitation signal + applied power.
    axB = axes[0, 1]
    _dark(axB)
    ts = timeseries
    axB.fill_between(ts["t"], 0, ts["cavitation_signal"], color="#ffae42",
                     alpha=0.7, label="Cavitation signal")
    axB2 = axB.twinx()
    axB2.plot(ts["t"], ts["power_pct"], color="#39ff14", lw=1.4, label="Applied power %")
    axB2.set_ylim(0, 105)
    axB2.tick_params(colors="#39ff14", labelsize=8)
    axB2.set_ylabel("Applied power (%)", color="#39ff14")
    axB.set_xlabel("Sonication time (s)")
    axB.set_ylabel("Cavitation signal (a.u.)", color="#ffae42")
    axB.set_title("(B) Acoustic controls — live feedback\n"
                  "cavitation emission (orange) and applied power (green)",
                  fontsize=9)

    # (C) Cavitation dose: cumulative toward goal.
    axC = axes[1, 0]
    _dark(axC)
    axC.plot(ts["t"], ts["cumulative_dose"], color="#ffae42", lw=1.8)
    axC.fill_between(ts["t"], 0, ts["cumulative_dose"], color="#ffae42", alpha=0.3)
    axC.axhline(ts["goal"], color="#ffff33", lw=1.2, ls="--", label="Prescribed dose goal")
    reach = np.argmax(ts["cumulative_dose"] >= ts["goal"])
    if ts["cumulative_dose"][reach] >= ts["goal"]:
        axC.axvline(ts["t"][reach], color="#ffff33", lw=0.8, ls=":")
        axC.text(ts["t"][reach], ts["goal"] * 0.4,
                 f" goal at {ts['t'][reach]:.0f}s", color="#ffff33", fontsize=8)
    axC.set_xlabel("Sonication time (s)")
    axC.set_ylabel("Cumulative cavitation dose (a.u.)")
    axC.set_title("(C) Cavitation dose — cumulative to goal", fontsize=9)
    axC.legend(loc="upper left", fontsize=7, facecolor="black",
               edgecolor="#888", labelcolor="#ddd")

    # (D) Per-spot cavitation dose distribution across the steered raster.
    axD = axes[1, 1]
    cmap = LinearSegmentedColormap.from_list(
        "cav", ["#101010", "#7a1500", "#ffae42", "#ffff99"])
    dose = spot["dose"]
    extent = [spot["lateral_mm"].min(), spot["lateral_mm"].max(),
              spot["axial_mm"].min(), spot["axial_mm"].max()]
    im = axD.imshow(dose, origin="lower", extent=extent, aspect="auto",
                    cmap=cmap)
    # Prescribed-dose contour: spots inside it reach the treatable goal.
    goal_level = float(spot.get("goal", 0.5 * float(dose.max())))
    if dose.max() > goal_level > 0:
        axD.contour(dose, levels=[goal_level], colors=["#ffff33"],
                    linewidths=1.2, extent=extent)
    pct = 100.0 * float((dose >= goal_level).mean())
    axD.set_xlabel("Lateral steering offset (mm)")
    axD.set_ylabel("Axial steering offset (mm)")
    axD.set_title(f"(D) Per-subspot dose vs electronic steering + attenuation\n"
                  f"yellow = treatable envelope (≥ goal); {pct:.0f}% of raster reaches goal",
                  fontsize=9)
    cb = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
    cb.set_label("Per-spot cavitation dose (a.u.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if modality_note:
        fig.text(0.5, 0.005, modality_note, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig
