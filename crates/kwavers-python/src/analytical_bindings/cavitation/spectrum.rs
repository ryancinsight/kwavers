//! Passive cavitation spectrum and dose PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compute the power spectrum of a bubble radius time series.
///
/// Args:
///     r_arr: Radius time series [m].
///     dt_s: Sample interval [s].
///     n_fft: FFT length.
///
/// Returns:
///     (frequencies [Hz], power spectral density) tuple.
#[pyfunction]
#[pyo3(signature = (r_arr, dt_s, n_fft))]
pub fn bubble_power_spectrum(
    py: Python<'_>,
    r_arr: PyReadonlyArray1<f64>,
    dt_s: f64,
    n_fft: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let r_s = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (freqs, psd) = cavitation::bubble_power_spectrum(r_s, dt_s, n_fft);
    Ok((
        freqs.into_pyarray(py).unbind(),
        psd.into_pyarray(py).unbind(),
    ))
}

/// Hann-windowed single-sided power spectral density of an emission series.
///
/// Suppresses spectral leakage from dominant harmonic lines so the inter-line
/// floor reflects true inharmonic (broadband / inertial) emission — the
/// estimator passive-cavitation-dose decomposition should run on.
///
/// Args:
///     signal: Emission time series (e.g. from bubble_acoustic_emission_pressure).
///     dt_s: Sample interval [s].
///     n_fft: FFT length (>= signal length; zero-padded).
///
/// Returns:
///     (frequencies [Hz], PSD) tuple over non-negative frequencies.
///
/// Reference:
///     Gyongy & Coussios (2010) JASA 128, 2403.
#[pyfunction]
#[pyo3(signature = (signal, dt_s, n_fft))]
pub fn hann_windowed_power_spectrum(
    py: Python<'_>,
    signal: PyReadonlyArray1<f64>,
    dt_s: f64,
    n_fft: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let s = signal
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (freqs, psd) = cavitation::hann_windowed_power_spectrum(s, dt_s, n_fft);
    Ok((
        freqs.into_pyarray(py).unbind(),
        psd.into_pyarray(py).unbind(),
    ))
}

/// Compute a normalized Keller-Miksis PCD spectrum and SC/IC band ratios.
#[pyfunction]
#[pyo3(signature = (
    r0_m,
    p_ac_pa,
    drive_frequency_hz,
    n_cycles,
    n_per_cycle,
    discard_cycles,
    p0_pa,
    rho,
    sigma,
    mu,
    kappa,
    vapor_pressure_pa,
    sound_speed_m_s,
))]
#[allow(clippy::too_many_arguments)]
pub fn keller_miksis_pcd_spectrum<'py>(
    py: Python<'py>,
    r0_m: f64,
    p_ac_pa: f64,
    drive_frequency_hz: f64,
    n_cycles: usize,
    n_per_cycle: usize,
    discard_cycles: usize,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    vapor_pressure_pa: f64,
    sound_speed_m_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let spectrum = cavitation::keller_miksis_pcd_spectrum(
        r0_m,
        p_ac_pa,
        drive_frequency_hz,
        n_cycles,
        n_per_cycle,
        discard_cycles,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        vapor_pressure_pa,
        sound_speed_m_s,
    )
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("frequency_hz", spectrum.frequency_hz.into_pyarray(py))?;
    out.set_item(
        "normalized_psd_db",
        spectrum.normalized_psd_db.into_pyarray(py),
    )?;
    out.set_item("stable_signal", spectrum.stable_signal)?;
    out.set_item("inertial_signal", spectrum.inertial_signal)?;
    Ok(out)
}

/// Compute a Keller-Miksis PCD feedback-controller trace.
#[pyfunction]
#[pyo3(signature = (
    r0_m,
    drive_frequency_hz,
    n_pulses,
    initial_pressure_pa,
    n_cycles_per_pulse,
    n_per_cycle,
    discard_cycles,
    stable_target,
    inertial_limit,
    gamma_up,
    gamma_down,
    p_min_pa,
    p_max_pa,
    p0_pa,
    rho,
    sigma,
    mu,
    kappa,
    vapor_pressure_pa,
    sound_speed_m_s,
))]
#[allow(clippy::too_many_arguments)]
pub fn keller_miksis_pcd_controller_trace<'py>(
    py: Python<'py>,
    r0_m: f64,
    drive_frequency_hz: f64,
    n_pulses: usize,
    initial_pressure_pa: f64,
    n_cycles_per_pulse: usize,
    n_per_cycle: usize,
    discard_cycles: usize,
    stable_target: f64,
    inertial_limit: f64,
    gamma_up: f64,
    gamma_down: f64,
    p_min_pa: f64,
    p_max_pa: f64,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    vapor_pressure_pa: f64,
    sound_speed_m_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let trace = cavitation::keller_miksis_pcd_controller_trace(
        r0_m,
        drive_frequency_hz,
        n_pulses,
        initial_pressure_pa,
        n_cycles_per_pulse,
        n_per_cycle,
        discard_cycles,
        stable_target,
        inertial_limit,
        gamma_up,
        gamma_down,
        p_min_pa,
        p_max_pa,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        vapor_pressure_pa,
        sound_speed_m_s,
    )
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("pulse_index", trace.pulse_index.into_pyarray(py))?;
    out.set_item("pressure_kpa", trace.pressure_kpa.into_pyarray(py))?;
    out.set_item("stable_signal", trace.stable_signal.into_pyarray(py))?;
    out.set_item("inertial_signal", trace.inertial_signal.into_pyarray(py))?;
    out.set_item(
        "stable_signal_normalized",
        trace.stable_signal_normalized.into_pyarray(py),
    )?;
    out.set_item(
        "inertial_signal_normalized",
        trace.inertial_signal_normalized.into_pyarray(py),
    )?;
    Ok(out)
}

/// Far-field acoustic emission pressure radiated by a pulsating bubble.
///
/// p_sc(r_obs, t) = (rho * R / r_obs) * (2*Rdot^2 + R*Rddot)
///
/// This is the signal a passive cavitation detector records, computed from the
/// radius/wall-velocity history returned by `solve_keller_miksis` /
/// `solve_rayleigh_plesset`. Rddot is obtained by central differences of Rdot.
///
/// Args:
///     r_arr: Bubble radius series R(t) [m].
///     rdot_arr: Wall velocity series Rdot(t) [m/s] (same length as r_arr).
///     dt_s: Uniform time step [s].
///     rho: Liquid density [kg/m³].
///     r_obs_m: Observation distance from the bubble [m].
///
/// Returns:
///     Emitted-pressure series p_sc(t) [Pa] (same length as r_arr).
///
/// Reference:
///     Leighton (1994) The Acoustic Bubble, §3.2.1; Neppiras (1980) Phys. Rep. 61, 159.
#[pyfunction]
#[pyo3(signature = (r_arr, rdot_arr, dt_s, rho, r_obs_m))]
pub fn bubble_acoustic_emission_pressure(
    py: Python<'_>,
    r_arr: PyReadonlyArray1<f64>,
    rdot_arr: PyReadonlyArray1<f64>,
    dt_s: f64,
    rho: f64,
    r_obs_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let r_s = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rd_s = rdot_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::bubble_acoustic_emission_pressure(r_s, rd_s, dt_s, rho, r_obs_m);
    Ok(result.into_pyarray(py).unbind())
}

/// Coherently superpose a microbubble population's emission series into one
/// ensemble time series.
///
/// y[t] = sum_i gains[i] * emissions[i][t - delays[i]]. Each of the n_bubbles
/// per-bubble series (rows of the (n_bubbles, n_samples) matrix) is placed at an
/// integer sample delay and gain, accumulating into a buffer of length out_len.
/// Genuine broadband emission is this ensemble effect: a single steady-state
/// bubble is a line spectrum, but a polydisperse population of transient
/// emissions at random nucleation delays fills the inter-harmonic floor.
/// Feed the result to `hann_windowed_power_spectrum`.
///
/// Args:
///     emissions: (n_bubbles, n_samples) per-bubble emission series.
///     delays_samples: per-bubble nucleation/arrival delay [samples].
///     gains: per-bubble amplitude weight.
///     out_len: length of the summed output buffer (>= n_samples + max delay).
///
/// Returns:
///     Summed ensemble emission series of length out_len.
///
/// Reference:
///     Gyongy & Coussios (2010) JASA 128, 2403.
#[pyfunction]
#[pyo3(signature = (emissions, delays_samples, gains, out_len))]
pub fn ensemble_emission_superposition(
    py: Python<'_>,
    emissions: PyReadonlyArray2<f64>,
    delays_samples: PyReadonlyArray1<i64>,
    gains: PyReadonlyArray1<f64>,
    out_len: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = emissions.as_array();
    let (n_bubbles, n_samples) = arr.dim();
    let flat: Vec<f64> = arr.iter().copied().collect();
    let delays: Vec<usize> = delays_samples
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .iter()
        .map(|&d| d.max(0) as usize)
        .collect();
    let g = gains
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::ensemble_emission_superposition(
        &flat, n_bubbles, n_samples, &delays, g, out_len,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// Decompose a passive-cavitation emission spectrum into cavitation bands.
///
/// Each PSD bin is assigned to the nearest half-harmonic line k*f0/2 within the
/// half-window rel_halfwidth*f0, integrated above the noise floor:
///   k even (>=2) -> harmonic comb (fundamental); k=1 -> subharmonic;
///   k odd (>=3) -> ultraharmonic; otherwise -> broadband.
///
/// Args:
///     freqs: Frequency axis [Hz], uniformly spaced ascending.
///     psd: Power spectral density at each frequency (same length).
///     f0_hz: Fundamental drive frequency [Hz].
///     rel_halfwidth: Line half-window as fraction of f0 (clamped to (0, 0.25)).
///     noise_floor: Baseline PSD subtracted from every bin (>= 0).
///
/// Returns:
///     (fundamental, subharmonic, ultraharmonic, broadband) band energies
///     [PSD-units * Hz]. Stable-cavitation emission = subharmonic + ultraharmonic;
///     inertial-cavitation emission = broadband.
///
/// Reference:
///     Gyongy & Coussios (2010) JASA 128, 2403; Arvanitis et al. (2012) PLoS ONE 7, e45783.
#[pyfunction]
#[pyo3(signature = (freqs, psd, f0_hz, rel_halfwidth, noise_floor))]
pub fn cavitation_emission_bands(
    freqs: PyReadonlyArray1<f64>,
    psd: PyReadonlyArray1<f64>,
    f0_hz: f64,
    rel_halfwidth: f64,
    noise_floor: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let f_s = freqs
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let p_s = psd
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let b = cavitation::decompose_emission_spectrum(f_s, p_s, f0_hz, rel_halfwidth, noise_floor);
    Ok((b.fundamental, b.subharmonic, b.ultraharmonic, b.broadband))
}

/// Normalized passive cavitation emission spectrum for stable or inertial regimes.
///
/// Args:
///     freqs_hz: Frequency axis [Hz].
///     f0_hz: Fundamental drive frequency [Hz].
///     regime: "stable" or "inertial".
///     snr_db: Signal-to-noise ratio used to add the finite noise floor [dB].
///
/// Returns:
///     Linear PSD normalized so its peak bin is 1.0.
#[pyfunction]
#[pyo3(signature = (freqs_hz, f0_hz, regime, snr_db=30.0))]
pub fn normalized_cavitation_emission_spectrum(
    py: Python<'_>,
    freqs_hz: PyReadonlyArray1<f64>,
    f0_hz: f64,
    regime: &str,
    snr_db: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let freqs = freqs_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let regime = match regime {
        "stable" => cavitation::CavitationEmissionRegime::Stable,
        "inertial" => cavitation::CavitationEmissionRegime::Inertial,
        other => {
            return Err(PyValueError::new_err(format!(
                "regime must be 'stable' or 'inertial', got '{other}'"
            )))
        }
    };
    let result = cavitation::normalized_cavitation_emission_spectrum(freqs, f0_hz, regime, snr_db);
    Ok(result.into_pyarray(py).unbind())
}

/// Cumulative cavitation dose: trapezoidal time-integral of an emission-power series.
///
/// D[m] = sum_{i=1..m} 0.5*(P[i-1]+P[i])*dt   [emission-power * s]
///
/// Feed the stable emission (sub+ultra) for the stable-cavitation dose, or the
/// broadband emission for the inertial-cavitation dose. Negative samples clamp to 0.
///
/// Args:
///     power_arr: Per-window band emission power.
///     dt_s: Monitoring-window duration [s].
///
/// Returns:
///     Running cumulative dose array (same length; D[0] = 0).
///
/// Reference:
///     O'Reilly & Hynynen (2012) Radiology 263, 96.
#[pyfunction]
#[pyo3(signature = (power_arr, dt_s))]
pub fn cumulative_cavitation_dose(
    py: Python<'_>,
    power_arr: PyReadonlyArray1<f64>,
    dt_s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = power_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::cumulative_cavitation_dose(p_s, dt_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Chapter 23 passive-cavitation dose traces.
///
/// Rust owns the deterministic stable-dose staircase and seeded compound
/// Poisson inertial-dose trials. Python receives arrays for plotting only.
#[pyfunction]
#[pyo3(signature = (time_s, prf_hz, pulse_duration_s, inertial_event_rate_fraction=0.3, seed=0))]
pub fn passive_cavitation_dose_fixture<'py>(
    py: Python<'py>,
    time_s: PyReadonlyArray1<f64>,
    prf_hz: f64,
    pulse_duration_s: f64,
    inertial_event_rate_fraction: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let time = time_s
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let trace = cavitation::passive_cavitation_dose_fixture(
        time,
        prf_hz,
        pulse_duration_s,
        inertial_event_rate_fraction,
        seed,
    )
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("time_s", trace.time_s.into_pyarray(py))?;
    out.set_item("stable_dose", trace.stable_dose.into_pyarray(py))?;
    out.set_item(
        "inertial_trial1_dose",
        trace.inertial_trial1_dose.into_pyarray(py),
    )?;
    out.set_item(
        "inertial_trial2_dose",
        trace.inertial_trial2_dose.into_pyarray(py),
    )?;
    Ok(out)
}
