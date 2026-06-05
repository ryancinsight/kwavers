"""Shared orchestration for passive-cavitation harmonic-dose monitoring.

Used by ch24 (BBB LIFU) and ch21e (liver histotripsy). This module is pure
glue: every physics step is a call into the kwavers Rust core
(`solve_keller_miksis`, `bubble_acoustic_emission_pressure`,
`bubble_power_spectrum`, `integrate_receiver_array_psd`,
`cavitation_emission_bands`, `cumulative_cavitation_dose`,
`cavitation_controller_pressure`). No domain math lives here — the only
Python-side numeric step is decimating the fine Keller–Miksis emission series
before the O(N²) reference DFT, which is sampling, not physics.

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


def _decimate(x: np.ndarray, n_target: int) -> tuple[np.ndarray, int]:
    """Return (decimated x, stride). Slicing only — no filtering/physics."""
    stride = max(1, x.size // max(n_target, 1))
    return np.ascontiguousarray(x[::stride]), stride


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
    t_end = n_cycles / f0
    n_steps = int(round(n_cycles * steps_per_cycle))
    channel_psds = []
    freqs_dec = None
    for r0 in np.atleast_1d(r0_population_m):
        _t, r_m, rdot = kw.solve_keller_miksis(
            float(r0), 0.0, medium.p0, drive_pa, f0, t_end, n_steps,
            medium.rho, medium.sigma, medium.gamma, medium.mu, medium.pv,
            medium.c_l, medium.xi_s,
        )
        r_m = np.asarray(r_m, dtype=float)
        rdot = np.asarray(rdot, dtype=float)
        if not (np.all(np.isfinite(r_m)) and np.all(np.isfinite(rdot))):
            # Keller–Miksis went stiff/unstable at this amplitude for this R0;
            # skip the bubble rather than poison the population spectrum.
            continue
        dt = t_end / n_steps
        emit = np.asarray(
            kw.bubble_acoustic_emission_pressure(r_m, rdot, dt, medium.rho, r_obs_m),
            dtype=float,
        )
        if emit.size == 0 or not np.all(np.isfinite(emit)):
            continue
        # Drop the ring-up transient and keep the steady-state tail so the
        # reference (rectangular-window) DFT sees a (near-)periodic record and
        # broadband reflects true inharmonic emission, not startup leakage.
        i0 = int(transient_fraction * emit.size)
        emit_ss = emit[i0:]
        emit_dec, stride = _decimate(emit_ss, n_fft)
        dt_dec = dt * stride
        # Hann-windowed PSD: suppresses harmonic-line leakage so the broadband
        # (inertial) band is genuine inharmonic emission, not DFT skirts.
        f, psd = kw.hann_windowed_power_spectrum(emit_dec, dt_dec, emit_dec.size)
        freqs_dec = np.asarray(f, dtype=float)
        channel_psds.append(np.asarray(psd, dtype=float))

    if not channel_psds or freqs_dec is None:
        return np.zeros(1), np.zeros(1)

    # Pad to a common length and incoherently power-sum across the population
    # (receiver-array integration over V_s).
    n_bins = min(p.size for p in channel_psds)
    stacked = np.array([p[:n_bins] for p in channel_psds])
    psd_vs = np.asarray(kw.integrate_receiver_array_psd(stacked), dtype=float)
    return freqs_dec[:n_bins], psd_vs


def simulate_population_emission(
    kw,
    *,
    drive_pa: float,
    f0: float,
    medium: BubbleMedium,
    n_bubbles: int,
    rng,
    r0_median_m: float = 1.5e-6,
    r0_sigma_ln: float = 0.4,
    n_cycles: float = 12.0,
    n_out: int = 8192,
    max_nucleation_cycles: float = 2.0,
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

    ``rng`` is a ``numpy.random.Generator`` (seed it for reproducible figures).
    Returns dict: ``freqs``, ``psd``, ``fundamental``, ``subharmonic``,
    ``ultraharmonic``, ``broadband``, ``stable`` (sub+ultra), ``total``,
    ``n_active``, ``max_compression``, ``max_mach``.
    """
    t_end = n_cycles / f0
    traces = []
    max_c = 0.0
    max_m = 0.0
    for _ in range(n_bubbles):
        r0 = float(rng.lognormal(np.log(r0_median_m), r0_sigma_ln))
        if coated:
            _t, _r, _rd, emit, mc, mm, _ncol, _conv = kw.simulate_coated_bubble_emission(
                r0, drive_pa, f0, n_cycles, n_out, r_obs_m,
                chi=chi, shell_viscosity=shell_viscosity, shell_thickness=shell_thickness,
                sigma_initial=sigma_initial, steps_per_cycle=steps_per_cycle,
                p0_pa=medium.p0, rho=medium.rho, c_liquid=medium.c_l, mu=medium.mu,
                gamma=medium.gamma)
        else:
            _t, _r, _rd, emit, mc, mm, _ncol, _conv = kw.simulate_bubble_emission(
                r0, drive_pa, f0, n_cycles, n_out, r_obs_m,
                p0_pa=medium.p0, rho=medium.rho, c_liquid=medium.c_l, mu=medium.mu,
                sigma=medium.sigma, pv=medium.pv, gamma=medium.gamma,
                thermal_effects=thermal_effects)
        emit = np.asarray(emit, dtype=float)
        # Reject non-finite traces and unphysical spikes (a fixed-step collapse
        # can emit a near-singular value just before the radius bound truncates
        # it — a far-field emission at r_obs ≫ R0 cannot exceed ~10 MPa).
        if emit.size < 16 or not np.all(np.isfinite(emit)) or np.abs(emit).max() > 1.0e7:
            continue
        traces.append(emit)
        max_c = max(max_c, float(mc))
        max_m = max(max_m, float(mm))

    empty = {"freqs": np.zeros(1), "psd": np.zeros(1), "fundamental": 0.0,
             "subharmonic": 0.0, "ultraharmonic": 0.0, "broadband": 0.0,
             "stable": 0.0, "total": 0.0, "n_active": 0,
             "max_compression": 0.0, "max_mach": 0.0}
    if not traces:
        return empty

    # The fully-converged length spans exactly n_cycles whole periods → the
    # fundamental lands ON an FFT bin (no leakage). Pad truncated-collapse traces
    # to that length (they emit, then go silent) so all share one bin grid.
    full_len = max(t.size for t in traces)
    dt_emit = t_end / (full_len - 1)
    freqs = np.fft.rfftfreq(full_len, dt_emit)
    win = np.hanning(full_len)
    psd_sum = np.zeros(freqs.size)
    for emit in traces:
        if emit.size < full_len:
            emit = np.concatenate([emit, np.zeros(full_len - emit.size)])
        # Incoherent power summation across the focal population (asynchronous
        # bubbles → powers add): keeps each band at its true single-bubble level,
        # broadband only from genuinely inertial collapse — no coherent-sum
        # cross-term artifacts.
        spec = np.fft.rfft((emit - emit.mean()) * win)
        psd_sum = psd_sum + np.abs(spec) ** 2
    n_active = len(traces)

    f = np.ascontiguousarray(freqs)
    psd = np.ascontiguousarray(psd_sum)
    fund, sub, ultra, broad = kw.cavitation_emission_bands(
        f, psd, f0, rel_halfwidth, noise_floor)
    return {
        "freqs": f, "psd": psd,
        "fundamental": fund, "subharmonic": sub, "ultraharmonic": ultra,
        "broadband": broad, "stable": sub + ultra,
        "total": fund + sub + ultra + broad, "n_active": n_active,
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
    keys = ("harmonic", "subharmonic", "ultraharmonic", "stable", "inertial", "signal")
    out = {k: np.zeros(pressures_pa.size) for k in keys}
    rng = np.random.default_rng(seed)
    for i, pa in enumerate(pressures_pa):
        r = simulate_population_emission(
            kw, drive_pa=float(pa), f0=f0, medium=medium,
            n_bubbles=n_bubbles, rng=rng, **sim_kwargs)
        out["harmonic"][i] = r["fundamental"]
        out["subharmonic"][i] = r["subharmonic"]
        out["ultraharmonic"][i] = r["ultraharmonic"]
        out["stable"][i] = r["stable"]
        out["inertial"][i] = r["broadband"]
        out["signal"][i] = r["stable"] + r["broadband"]
    return out


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
    out = {k: np.zeros(pressures_pa.size)
           for k in ("harmonic", "subharmonic", "ultraharmonic",
                     "stable", "inertial")}
    for i, pa in enumerate(pressures_pa):
        freqs, psd = vs_emission_spectrum(
            kw, drive_pa=float(pa), f0=f0,
            r0_population_m=r0_population_m, medium=medium,
            **spectrum_kwargs,
        )
        fund, sub, ultra, broad = kw.cavitation_emission_bands(
            freqs, psd, f0, rel_halfwidth, noise_floor,
        )
        out["harmonic"][i] = fund
        out["subharmonic"][i] = sub
        out["ultraharmonic"][i] = ultra
        out["stable"][i] = sub + ultra
        out["inertial"][i] = broad
    return out


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
    p_lo, p_hi = float(pressures_pa.min()), float(pressures_pa.max())

    p = float(np.clip(p_start_pa, p_lo, p_hi))
    p_hist = np.zeros(n_bursts)
    se_hist = np.zeros(n_bursts)
    ie_hist = np.zeros(n_bursts)
    for k in range(n_bursts):
        se = float(np.interp(p, pressures_pa, stable_power))
        ie = float(np.interp(p, pressures_pa, inertial_power))
        p_hist[k] = p
        se_hist[k] = se
        ie_hist[k] = ie
        p = kw.cavitation_controller_pressure(
            p, se, ie, stable_target, inertial_limit, gain, p_lo, p_hi,
        )

    return {
        "pressure": p_hist,
        "stable_emission": se_hist,
        "inertial_emission": ie_hist,
        "stable_dose": np.asarray(
            kw.cumulative_cavitation_dose(se_hist, burst_duration_s), dtype=float),
        "inertial_dose": np.asarray(
            kw.cumulative_cavitation_dose(ie_hist, burst_duration_s), dtype=float),
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
    p_lo, p_hi = float(pressures_pa.min()), float(pressures_pa.max())
    rng = np.random.default_rng(seed)
    dt = 1.0 / prf_hz

    p = float(np.clip(p_start_pa, p_lo, p_hi))
    sig = np.zeros(n_pulses)
    pwr = np.zeros(n_pulses)
    for k in range(n_pulses):
        det = float(np.interp(p, pressures_pa, cavitation_power))
        measured = det * float(rng.lognormal(mean=0.0, sigma=jitter_sigma))
        sig[k] = measured
        pwr[k] = (p / p_hi) ** 2 * 100.0  # acoustic power ∝ p²
        # Controller reacts to the measured signal: cap inertial, else recruit.
        p = kw.cavitation_controller_pressure(
            p, measured, measured, target_signal, inertial_cap, gain, p_lo, p_hi)

    t = np.arange(n_pulses) * dt
    cumulative = np.asarray(kw.cumulative_cavitation_dose(sig, dt), dtype=float)
    goal = goal_fraction * float(cumulative[-1]) if cumulative[-1] > 0 else 1.0
    return {
        "t": t,
        "cavitation_signal": sig,
        "power_pct": pwr,
        "cumulative_dose": cumulative,
        "goal": goal,
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
    rng = np.random.default_rng(seed)
    dt = 1.0 / prf_hz
    p = float(np.clip(p_start_pa, p_lo_pa, p_hi_pa))
    sig = np.zeros(n_pulses)
    pwr = np.zeros(n_pulses)
    stable = np.zeros(n_pulses)
    broad = np.zeros(n_pulses)
    for k in range(n_pulses):
        r = simulate_population_emission(
            kw, drive_pa=p, f0=f0, medium=medium, n_bubbles=n_bubbles,
            rng=rng, **sim_kwargs)
        s_st, s_br = r["stable"], r["broadband"]
        stable[k] = s_st
        broad[k] = s_br
        sig[k] = s_st + s_br
        pwr[k] = (p / p_hi_pa) ** 2 * 100.0
        p = kw.cavitation_controller_pressure(
            p, s_st, s_br, target_signal, inertial_cap, gain, p_lo_pa, p_hi_pa)

    t = np.arange(n_pulses) * dt
    cumulative = np.asarray(kw.cumulative_cavitation_dose(sig, dt), dtype=float)
    goal = goal_fraction * float(cumulative[-1]) if cumulative[-1] > 0 else 1.0
    return {
        "t": t, "cavitation_signal": sig, "power_pct": pwr,
        "cumulative_dose": cumulative, "goal": goal,
        "stable_signal": stable, "broadband_signal": broad,
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

    Each subspot's delivered peak pressure is set by the electronic-steering
    efficiency (Rust ``electronic_steering_efficiency``) and tissue attenuation.
    The firing order is then either:
      * ``"sequential"`` — all ``pulses_per_spot`` pulses at spot 0, then spot 1,
        …; consecutive pulses at a spot are 1/PRF apart.
      * ``"interleaved"`` — round-robin over the ``interleave_group`` (default:
        all spots); consecutive pulses at a spot are ``group/PRF`` apart, so the
        spot rests while the others fire.

    Two physical memory effects make the orders differ — both emergent from the
    per-spot inter-pulse interval Δt:
      * residual-bubble shielding (Macoskey 2018), via the Rust
        ``prf_efficacy_factor`` at the effective per-spot PRF = 1/Δt — short Δt
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
    n_spots = lat.size
    cav_pressures_pa = np.asarray(cav_pressures_pa, float)
    cav_dose_per_pulse = np.asarray(cav_dose_per_pulse, float)

    # Per-spot delivered pressure and per-pulse base cavitation dose.
    eps = np.array([
        kw.electronic_steering_efficiency(float(lat[i]), float(ax[i]), f0_hz, c_m_s, apodized)
        for i in range(n_spots)])
    trans = np.exp(-attenuation_np_m * np.maximum(ax, 0.0))
    p_spot = p_target_pa * eps * trans
    base_dose = np.interp(p_spot, cav_pressures_pa, cav_dose_per_pulse)

    group = n_spots if interleave_group <= 0 else min(interleave_group, n_spots)
    if schedule == "sequential":
        dt_spot = 1.0 / prf_hz
        order = np.repeat(np.arange(n_spots), pulses_per_spot)
    else:  # interleaved round-robin
        dt_spot = group / prf_hz
        order = np.tile(np.arange(n_spots), pulses_per_spot)

    # Residual-bubble shielding efficacy at the steady per-spot interval (Rust).
    eff_steady = float(kw.prf_efficacy_factor(
        np.array([1.0 / dt_spot]), tau_dissolution_s, shielding_g)[0])

    decay = float(np.exp(-dt_spot / max(tau_thermal_s, 1e-12)))
    dT_pulse = thermal_gain_k_per_pulse * (p_spot / max(p_target_pa, 1.0)) ** 2

    per_spot_dose = np.zeros(n_spots)
    per_spot_T = np.zeros(n_spots)         # temperature rise over baseline
    per_spot_peak_T = np.zeros(n_spots)
    last_fire = np.full(n_spots, -np.inf)
    n_fire = order.size
    t_axis = np.zeros(n_fire)
    cum_dose = np.zeros(n_fire)
    coverage = np.zeros(n_fire)
    running = 0.0
    for k, s in enumerate(order):
        t = k / prf_hz
        gap = t - last_fire[s]
        # First pulse at a spot sees no residual; later pulses see shielding.
        eff = 1.0 if not np.isfinite(gap) else eff_steady
        per_spot_dose[s] += base_dose[s] * eff
        running += base_dose[s] * eff
        # Thermal relaxation since this spot last fired, then this pulse's rise.
        cool = 1.0 if not np.isfinite(gap) else float(np.exp(-gap / max(tau_thermal_s, 1e-12)))
        per_spot_T[s] = per_spot_T[s] * cool + dT_pulse[s]
        per_spot_peak_T[s] = max(per_spot_peak_T[s], per_spot_T[s])
        last_fire[s] = t
        t_axis[k] = t
        cum_dose[k] = running
        coverage[k] = float(np.mean(per_spot_dose >= goal_dose)) if goal_dose > 0 else 0.0

    # Resample to a compact uniform time grid for plotting.
    treatment_s = float(t_axis[-1]) if n_fire else 0.0
    tq = np.linspace(0.0, max(treatment_s, 1e-9), n_time_samples)
    return {
        "time": tq,
        "coverage": np.interp(tq, t_axis, coverage) if n_fire else tq * 0,
        "cumulative_dose": np.interp(tq, t_axis, cum_dose) if n_fire else tq * 0,
        "per_spot_dose": per_spot_dose,
        "per_spot_peak_temp": per_spot_peak_T,
        "efficacy": eff_steady,
        "dt_spot_s": dt_spot,
        "treatment_s": treatment_s,
        "lateral_mm": lat * 1e3,
        "axial_mm": ax * 1e3,
        "p_spot_pa": p_spot,
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

    dose = np.zeros((ax.size, lat.size))
    eff = np.zeros_like(dose)
    p_spot = np.zeros_like(dose)
    for i, dz in enumerate(ax):
        for j, dx in enumerate(lat):
            e = kw.electronic_steering_efficiency(float(dx), float(dz), f0_hz, c_m_s, apodized)
            trans = np.exp(-attenuation_np_m * max(float(dz), 0.0))
            p = p_target_pa * e * trans
            eff[i, j] = e
            p_spot[i, j] = p
            dose[i, j] = n_pulses_per_spot * float(
                np.interp(p, pressures_pa, cavitation_power))
    # Prescribed per-spot dose goal: the dose a spot receives at the minimum
    # effective drive pressure (defines the treatable steering envelope).
    goal = n_pulses_per_spot * float(
        np.interp(goal_pressure_pa, pressures_pa, cavitation_power))
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
