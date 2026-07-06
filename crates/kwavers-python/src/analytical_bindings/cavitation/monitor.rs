//! Passive cavitation monitoring and closed-loop control PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Curve-driven real-time passive-cavitation monitor trace.
///
/// Returns `(time_s, cavitation_signal, power_percent, cumulative_dose,
/// goal_dose)`. The Rust core performs pressure-curve interpolation, seeded
/// log-normal measurement jitter, controller stepping, acoustic-power scaling,
/// and cumulative dose integration.
#[pyfunction]
#[pyo3(signature = (pressures_pa, cavitation_power, n_pulses, prf_hz, p_start_pa, target_signal, inertial_cap, gain=0.06, jitter_sigma=0.35, goal_fraction=0.85, seed=0))]
pub fn cavitation_monitor_timeseries(
    py: Python<'_>,
    pressures_pa: PyReadonlyArray1<f64>,
    cavitation_power: PyReadonlyArray1<f64>,
    n_pulses: usize,
    prf_hz: f64,
    p_start_pa: f64,
    target_signal: f64,
    inertial_cap: f64,
    gain: f64,
    jitter_sigma: f64,
    goal_fraction: f64,
    seed: u64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
)> {
    let pressures = pressures_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let power = cavitation_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let trace = cavitation::cavitation_monitor_trace(cavitation::CavitationMonitorTraceInput {
        pressures_pa: pressures,
        cavitation_power: power,
        n_pulses,
        prf_hz,
        p_start_pa,
        target_signal,
        inertial_cap,
        gain,
        jitter_sigma,
        goal_fraction,
        seed,
    })
    .ok_or_else(|| PyRuntimeError::new_err("invalid cavitation monitor trace parameters"))?;

    Ok((
        trace.time_s.to_pyarray(py).unbind(),
        trace.cavitation_signal.to_pyarray(py).unbind(),
        trace.power_percent.to_pyarray(py).unbind(),
        trace.cumulative_dose.to_pyarray(py).unbind(),
        trace.goal_dose,
    ))
}

/// Fully simulated passive-cavitation monitor trace from bubble populations.
///
/// Returns `(time_s, cavitation_signal, power_percent, cumulative_dose,
/// goal_dose, stable_signal, broadband_signal)`. The Rust core performs the
/// per-pulse population-emission simulation, controller stepping, acoustic-power
/// scaling, and cumulative dose integration.
#[pyfunction]
#[pyo3(signature = (
    f0_hz, n_bubbles, n_pulses, prf_hz, p_start_pa, p_min_pa, p_max_pa,
    target_signal, inertial_cap, gain=0.06, goal_fraction=0.85, seed=0,
    r0_median_m=1.5e-6, r0_sigma_ln=0.4, n_cycles=12.0, n_out=8192,
    r_obs_m=5.0e-2, rel_halfwidth=0.12, noise_floor=0.0,
    thermal_effects=false, coated=false, chi=0.5, shell_viscosity=0.5,
    shell_thickness=3.0e-9, sigma_initial=0.04, steps_per_cycle=2000,
    p0_pa=101_325.0, rho=998.0, c_liquid=1481.0, mu=1.0e-3,
    sigma=0.0725, pv=2330.0, gamma=1.4
))]
#[allow(clippy::too_many_arguments)]
pub fn simulated_population_monitor_timeseries(
    py: Python<'_>,
    f0_hz: f64,
    n_bubbles: usize,
    n_pulses: usize,
    prf_hz: f64,
    p_start_pa: f64,
    p_min_pa: f64,
    p_max_pa: f64,
    target_signal: f64,
    inertial_cap: f64,
    gain: f64,
    goal_fraction: f64,
    seed: u64,
    r0_median_m: f64,
    r0_sigma_ln: f64,
    n_cycles: f64,
    n_out: usize,
    r_obs_m: f64,
    rel_halfwidth: f64,
    noise_floor: f64,
    thermal_effects: bool,
    coated: bool,
    chi: f64,
    shell_viscosity: f64,
    shell_thickness: f64,
    sigma_initial: f64,
    steps_per_cycle: usize,
    p0_pa: f64,
    rho: f64,
    c_liquid: f64,
    mu: f64,
    sigma: f64,
    pv: f64,
    gamma: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let trace = py
        .detach(|| {
            cavitation::simulated_population_monitor_trace(
                cavitation::SimulatedPopulationMonitorInput {
                    f0_hz,
                    medium: cavitation::PopulationMedium {
                        p0_pa,
                        rho,
                        c_liquid,
                        mu,
                        sigma,
                        pv,
                        gamma,
                    },
                    n_bubbles,
                    n_pulses,
                    prf_hz,
                    p_start_pa,
                    p_min_pa,
                    p_max_pa,
                    target_signal,
                    inertial_cap,
                    gain,
                    goal_fraction,
                    seed,
                    r0_median_m,
                    r0_sigma_ln,
                    n_cycles,
                    n_out,
                    r_obs_m,
                    rel_halfwidth,
                    noise_floor,
                    thermal_effects,
                    shell: cavitation::PopulationShell {
                        coated,
                        chi,
                        shell_viscosity,
                        shell_thickness,
                        sigma_initial,
                        steps_per_cycle,
                    },
                },
            )
        })
        .ok_or_else(|| {
            PyRuntimeError::new_err("invalid simulated population monitor parameters")
        })?;

    Ok((
        trace.time_s.to_pyarray(py).unbind(),
        trace.cavitation_signal.to_pyarray(py).unbind(),
        trace.power_percent.to_pyarray(py).unbind(),
        trace.cumulative_dose.to_pyarray(py).unbind(),
        trace.goal_dose,
        trace.stable_signal.to_pyarray(py).unbind(),
        trace.broadband_signal.to_pyarray(py).unbind(),
    ))
}

/// Closed-loop passive-cavitation sonication from swept stable/inertial curves.
///
/// Returns `(pressure, stable_emission, inertial_emission, stable_dose,
/// inertial_dose)`. The Rust core performs pressure-curve interpolation,
/// controller stepping, and stable/inertial dose accumulation.
#[pyfunction]
#[pyo3(signature = (pressures_pa, stable_power, inertial_power, n_bursts, burst_duration_s, p_start_pa, stable_target, inertial_limit, gain=0.08))]
pub fn closed_loop_cavitation_sonication(
    py: Python<'_>,
    pressures_pa: PyReadonlyArray1<f64>,
    stable_power: PyReadonlyArray1<f64>,
    inertial_power: PyReadonlyArray1<f64>,
    n_bursts: usize,
    burst_duration_s: f64,
    p_start_pa: f64,
    stable_target: f64,
    inertial_limit: f64,
    gain: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let pressures = pressures_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let stable = stable_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let inertial = inertial_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let trace = cavitation::closed_loop_cavitation_sonication(
        cavitation::ClosedLoopCavitationSonicationInput {
            pressures_pa: pressures,
            stable_power: stable,
            inertial_power: inertial,
            n_bursts,
            burst_duration_s,
            p_start_pa,
            stable_target,
            inertial_limit,
            gain,
        },
    )
    .ok_or_else(|| {
        PyRuntimeError::new_err("invalid closed-loop cavitation sonication parameters")
    })?;

    Ok((
        trace.pressure_pa.to_pyarray(py).unbind(),
        trace.stable_emission.to_pyarray(py).unbind(),
        trace.inertial_emission.to_pyarray(py).unbind(),
        trace.stable_dose.to_pyarray(py).unbind(),
        trace.inertial_dose.to_pyarray(py).unbind(),
    ))
}

/// Time-resolved cavitation dose for sequential or interleaved raster pulsing.
///
/// Returns `(time_s, coverage, cumulative_dose, per_spot_dose,
/// per_spot_peak_temp_k, efficacy, dt_spot_s, treatment_s, p_spot_pa)`. The Rust
/// core performs steering derating, attenuation, pressure-sweep interpolation,
/// firing-order selection, residual-bubble shielding, thermal relaxation, and
/// uniform resampling for plotting.
#[pyfunction]
#[pyo3(signature = (spot_lateral_m, spot_axial_m, p_target_pa, f0_hz, c_m_s, cav_pressures_pa, cav_dose_per_pulse, pulses_per_spot, prf_hz, schedule, interleave_group=0, attenuation_np_m=0.0, apodized=true, tau_dissolution_s=5.0e-3, shielding_g=1.2, tau_thermal_s=1.0, thermal_gain_k_per_pulse=0.0, goal_dose=0.0, n_time_samples=240))]
#[allow(clippy::too_many_arguments)]
pub fn raster_cavitation_pulsing(
    py: Python<'_>,
    spot_lateral_m: PyReadonlyArray1<f64>,
    spot_axial_m: PyReadonlyArray1<f64>,
    p_target_pa: f64,
    f0_hz: f64,
    c_m_s: f64,
    cav_pressures_pa: PyReadonlyArray1<f64>,
    cav_dose_per_pulse: PyReadonlyArray1<f64>,
    pulses_per_spot: usize,
    prf_hz: f64,
    schedule: &str,
    interleave_group: usize,
    attenuation_np_m: f64,
    apodized: bool,
    tau_dissolution_s: f64,
    shielding_g: f64,
    tau_thermal_s: f64,
    thermal_gain_k_per_pulse: f64,
    goal_dose: f64,
    n_time_samples: usize,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    Py<PyArray1<f64>>,
)> {
    let schedule = match schedule {
        "sequential" => cavitation::RasterPulsingSchedule::Sequential,
        "interleaved" => cavitation::RasterPulsingSchedule::Interleaved,
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "unknown raster pulsing schedule: {other}"
            )));
        }
    };
    let lateral = spot_lateral_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let axial = spot_axial_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pressures = cav_pressures_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dose = cav_dose_per_pulse
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let trace = cavitation::raster_cavitation_pulsing(cavitation::RasterPulsingInput {
        spot_lateral_m: lateral,
        spot_axial_m: axial,
        p_target_pa,
        f0_hz,
        c_m_s,
        cav_pressures_pa: pressures,
        cav_dose_per_pulse: dose,
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
    })
    .ok_or_else(|| PyRuntimeError::new_err("invalid raster cavitation pulsing parameters"))?;

    Ok((
        trace.time_s.to_pyarray(py).unbind(),
        trace.coverage.to_pyarray(py).unbind(),
        trace.cumulative_dose.to_pyarray(py).unbind(),
        trace.per_spot_dose.to_pyarray(py).unbind(),
        trace.per_spot_peak_temp_k.to_pyarray(py).unbind(),
        trace.efficacy,
        trace.dt_spot_s,
        trace.treatment_s,
        trace.p_spot_pa.to_pyarray(py).unbind(),
    ))
}

/// Classify passive-cavitation therapeutic-window indices from band powers.
///
/// Returns `(stable_onset_index, inertial_onset_index, controller_cap_index)`.
///
/// Args:
///     harmonic_power: Harmonic-comb emission power samples.
///     stable_power: Subharmonic + ultraharmonic emission power samples.
///     inertial_power: Broadband emission power samples.
///     stable_ratio_threshold: Stable/harmonic onset ratio.
///     inertial_ratio_threshold: Inertial/harmonic onset ratio.
///     cap_ratio_threshold: Conservative controller cap ratio.
///     denominator_floor: Positive floor added to harmonic power.
#[pyfunction]
#[pyo3(signature = (harmonic_power, stable_power, inertial_power, stable_ratio_threshold, inertial_ratio_threshold, cap_ratio_threshold, denominator_floor=1.0e-30))]
pub fn cavitation_therapeutic_window_indices(
    harmonic_power: PyReadonlyArray1<f64>,
    stable_power: PyReadonlyArray1<f64>,
    inertial_power: PyReadonlyArray1<f64>,
    stable_ratio_threshold: f64,
    inertial_ratio_threshold: f64,
    cap_ratio_threshold: f64,
    denominator_floor: f64,
) -> PyResult<(usize, usize, usize)> {
    let harmonic = harmonic_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let stable = stable_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let inertial = inertial_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let window = cavitation::cavitation_therapeutic_window_indices(
        harmonic,
        stable,
        inertial,
        stable_ratio_threshold,
        inertial_ratio_threshold,
        cap_ratio_threshold,
        denominator_floor,
    );
    Ok((
        window.stable_onset_index,
        window.inertial_onset_index,
        window.controller_cap_index,
    ))
}

/// First index where broadband emission exceeds a fraction of total emission.
///
/// Returns the first index where
/// `inertial / (harmonic + stable + inertial + denominator_floor) > threshold`,
/// with the documented Rust fallback and minimum-index clamp.
#[pyfunction]
#[pyo3(signature = (harmonic_power, stable_power, inertial_power, threshold, denominator_floor=1.0e-30, min_index=1))]
pub fn cavitation_inertial_fraction_onset_index(
    harmonic_power: PyReadonlyArray1<f64>,
    stable_power: PyReadonlyArray1<f64>,
    inertial_power: PyReadonlyArray1<f64>,
    threshold: f64,
    denominator_floor: f64,
    min_index: usize,
) -> PyResult<usize> {
    let harmonic = harmonic_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let stable = stable_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let inertial = inertial_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(cavitation::cavitation_inertial_fraction_onset_index(
        harmonic,
        stable,
        inertial,
        threshold,
        denominator_floor,
        min_index,
    ))
}

/// Rastered per-spot passive-cavitation dose across a steered treatment grid.
///
/// Returns row-major `(axial × lateral)` arrays `(dose, efficiency, p_spot_pa)`
/// plus the scalar `goal_dose`. The Rust core performs electronic-steering
/// derating, positive-axial attenuation, and endpoint-clamped interpolation over
/// the measured pressure/emission-power sweep.
#[pyfunction]
#[pyo3(signature = (lateral_offsets_m, axial_offsets_m, p_target_pa, f0_hz, c_m_s, pressures_pa, cavitation_power, n_pulses_per_spot, goal_pressure_pa, attenuation_np_m=0.0, apodized=true))]
pub fn per_spot_cavitation_dose_grid(
    py: Python<'_>,
    lateral_offsets_m: PyReadonlyArray1<f64>,
    axial_offsets_m: PyReadonlyArray1<f64>,
    p_target_pa: f64,
    f0_hz: f64,
    c_m_s: f64,
    pressures_pa: PyReadonlyArray1<f64>,
    cavitation_power: PyReadonlyArray1<f64>,
    n_pulses_per_spot: usize,
    goal_pressure_pa: f64,
    attenuation_np_m: f64,
    apodized: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, f64)> {
    let lateral = lateral_offsets_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let axial = axial_offsets_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pressures = pressures_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let power = cavitation_power
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let grid = cavitation::per_spot_cavitation_dose_grid(cavitation::PerSpotCavitationDoseInput {
        lateral_offsets_m: lateral,
        axial_offsets_m: axial,
        p_target_pa,
        f0_hz,
        c_m_s,
        pressures_pa: pressures,
        cavitation_power: power,
        n_pulses_per_spot,
        goal_pressure_pa,
        attenuation_np_m,
        apodized,
    })
    .ok_or_else(|| PyRuntimeError::new_err("invalid per-spot cavitation dose grid parameters"))?;

    Ok((
        grid.dose.to_pyarray(py).unbind(),
        grid.efficiency.to_pyarray(py).unbind(),
        grid.p_spot_pa.to_pyarray(py).unbind(),
        grid.goal_dose,
    ))
}

/// One step of the closed-loop cavitation-dose pressure controller.
///
/// Safety dominates: if broadband (inertial) emission exceeds inertial_limit,
/// pressure is reduced by (1-gain); else if stable (sub+ultra) emission is below
/// stable_target it is raised by (1+gain); otherwise held. Result clamped to
/// [p_min_pa, p_max_pa].
///
/// Args:
///     current_p_pa: Drive pressure on the just-monitored burst [Pa].
///     stable_emission: Measured sub+ultra-harmonic emission this burst.
///     inertial_emission: Measured broadband emission this burst.
///     stable_target: Stable-emission set-point.
///     inertial_limit: Broadband-emission ceiling.
///     gain: Fractional pressure step per burst (>= 0).
///     p_min_pa, p_max_pa: Drive-pressure clamp [Pa].
///
/// Returns:
///     Drive pressure for the next burst [Pa].
///
/// Reference:
///     McDannold et al. (2006) Phys. Med. Biol. 51, 793.
#[pyfunction]
#[pyo3(signature = (current_p_pa, stable_emission, inertial_emission, stable_target, inertial_limit, gain, p_min_pa, p_max_pa))]
#[allow(clippy::too_many_arguments)]
pub fn cavitation_controller_pressure(
    current_p_pa: f64,
    stable_emission: f64,
    inertial_emission: f64,
    stable_target: f64,
    inertial_limit: f64,
    gain: f64,
    p_min_pa: f64,
    p_max_pa: f64,
) -> PyResult<f64> {
    Ok(cavitation::cavitation_controller_pressure(
        current_p_pa,
        stable_emission,
        inertial_emission,
        stable_target,
        inertial_limit,
        gain,
        p_min_pa,
        p_max_pa,
    ))
}

