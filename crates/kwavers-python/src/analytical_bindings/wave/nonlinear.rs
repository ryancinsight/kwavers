//! PyO3 bindings: nonlinear wave physics (Fubini harmonics, shock formation,
//! tone burst, pulse train, Goldberg parameter, shock waveform,
//! Westervelt harmonic evolution).

use kwavers_physics::analytical::wave;
use leto::Array2;
use numpy::{ToPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Compute the n-th Fubini harmonic amplitude for a finite-amplitude wave.
///
/// Args:
///     n: Harmonic index (1-based).
///     sigma: Fubini variable (normalised propagation distance).
///
/// Returns:
///     Normalised harmonic amplitude.
#[pyfunction]
#[pyo3(signature = (n, sigma))]
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> PyResult<f64> {
    Ok(wave::fubini_harmonic_amplitude(n, sigma))
}

/// Compute the full Fubini harmonic spectrum up to order *n_max*.
///
/// Args:
///     n_max: Highest harmonic order to include.
///     sigma: Fubini variable.
///
/// Returns:
///     Array of normalised harmonic amplitudes, length *n_max*.
#[pyfunction]
#[pyo3(signature = (n_max, sigma))]
pub fn fubini_harmonic_spectrum(
    py: Python<'_>,
    n_max: u32,
    sigma: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = wave::fubini_harmonic_spectrum(n_max, sigma);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the shock formation distance for a finite-amplitude plane wave.
///
/// x_s = rho0 * c0^3 / (beta * omega * p0)
///
/// Args:
///     p0_pa: Source pressure amplitude [Pa].
///     f0_hz: Fundamental frequency [Hz].
///     c0: Small-signal sound speed [m/s].
///     rho0: Ambient density [kg/m³].
///     beta: Nonlinearity coefficient (1 + B/(2A)).
///
/// Returns:
///     Shock formation distance [m].
#[pyfunction]
#[pyo3(signature = (p0_pa, f0_hz, c0, rho0, beta))]
pub fn shock_formation_distance(
    p0_pa: f64,
    f0_hz: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
) -> PyResult<f64> {
    Ok(wave::shock_formation_distance(p0_pa, f0_hz, c0, rho0, beta))
}

/// Generate a Hann-windowed tone burst pressure waveform.
///
/// p(t) = A · w(t) · sin(2π·f₀·t), where w(t) = ½·(1 − cos(2π·t/τ)),  τ = n_cycles/f₀.
///
/// Args:
///     t_arr: Time sample points [s].
///     amplitude_pa: Peak pressure amplitude [Pa].
///     freq_hz: Carrier frequency [Hz].
///     n_cycles: Number of cycles in the burst (positive real).
///
/// Returns:
///     Pressure waveform [Pa], same length as t_arr.
///
/// Reference:
///     Harris (1978) Proc. IEEE 66, 51; k-Wave tone_burst convention.
#[pyfunction]
#[pyo3(signature = (t_arr, amplitude_pa, freq_hz, n_cycles))]
pub fn tone_burst_waveform(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    amplitude_pa: f64,
    freq_hz: f64,
    n_cycles: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::tone_burst_waveform(t_s, amplitude_pa, freq_hz, n_cycles);
    Ok(result.to_pyarray(py).unbind())
}

/// Generate a centered Hann-windowed tone burst on an existing time axis.
///
/// This binding is used by diagnostic PSF figures whose RF pulse is centered at
/// zero and whose Hann taper is defined by the number of selected samples.
#[pyfunction]
#[pyo3(signature = (t_arr, amplitude_pa, freq_hz, n_cycles))]
pub fn centered_hann_tone_burst_waveform(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    amplitude_pa: f64,
    freq_hz: f64,
    n_cycles: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::centered_hann_tone_burst_waveform(t_s, amplitude_pa, freq_hz, n_cycles);
    Ok(result.to_pyarray(py).unbind())
}

/// Superimpose Hann-windowed tone bursts at arbitrary start times (pulse train).
///
/// p(t) = sum_k A * w(t - t_k) * sin(2pi*f0*(t-t_k))
///
/// Args:
///     t_arr: Full time axis [s].
///     amplitude_pa: Per-burst peak amplitude [Pa].
///     freq_hz: Carrier frequency [Hz].
///     n_cycles: Cycles per burst.
///     t_starts: Burst start times [s].
///
/// Returns:
///     Accumulated pressure waveform [Pa], same length as t_arr.
///
/// Reference:
///     Harris (1978) Proc. IEEE 66, 51.
///     Macoskey et al. (2018) Ultrasound Med. Biol. 44, 2971.
#[pyfunction]
#[pyo3(signature = (t_arr, amplitude_pa, freq_hz, n_cycles, t_starts))]
pub fn pulse_train_waveform(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    amplitude_pa: f64,
    freq_hz: f64,
    n_cycles: f64,
    t_starts: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ts_s = t_starts
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::pulse_train_waveform(t_s, amplitude_pa, freq_hz, n_cycles, ts_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Goldberg shock parameter sweep over pulse durations.
///
/// sigma(tau) = beta * 2*pi*f * pnp * tau / (rho * c^2)
///
/// sigma < 1: pre-shock (Fubini valid); sigma = 1: shock onset;
/// sigma > 1: post-shock (Rankine-Hugoniot required).
///
/// Args:
///     pnp_pa: Peak negative pressure [Pa].
///     freq_hz: Fundamental frequency [Hz].
///     c: Sound speed [m/s].
///     rho: Density [kg/m3].
///     beta: Nonlinearity coefficient 1 + B/(2A).
///     tau_arr: Pulse durations [s].
///
/// Returns:
///     Goldberg parameter array, same length as tau_arr.
///
/// Reference:
///     Hamilton & Blackstock (1998) Nonlinear Acoustics, S3.3.
#[pyfunction]
#[pyo3(signature = (pnp_pa, freq_hz, c, rho, beta, tau_arr))]
pub fn goldberg_shock_parameter_sweep(
    py: Python<'_>,
    pnp_pa: f64,
    freq_hz: f64,
    c: f64,
    rho: f64,
    beta: f64,
    tau_arr: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let tau_s = tau_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::goldberg_shock_parameter_sweep(pnp_pa, freq_hz, c, rho, beta, tau_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Shock-enhanced absorption gain factor (phenomenological model).
///
/// G(sigma) = 1 + 9*sigma / (sigma + 1)
///
/// G -> 1 at sigma=0 (linear), G -> 10 at sigma->inf (fully shocked).
///
/// Args:
///     sigma_arr: Goldberg shock parameters (element-wise).
///
/// Returns:
///     Gain factor array, same length as sigma_arr.
///
/// Reference:
///     Hamilton & Blackstock (1998) Nonlinear Acoustics, S4.3.
#[pyfunction]
#[pyo3(signature = (sigma_arr,))]
pub fn shock_enhanced_absorption_gain(
    py: Python<'_>,
    sigma_arr: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let s_s = sigma_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::shock_enhanced_absorption_gain(s_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Effective pressure amplitude of a shock-distorted waveform.
///
/// p_eff(sigma) = pnp * (1 + (ppp/pnp - 1) * sigma / (sigma + 1))
///
/// Args:
///     pnp_pa: Peak negative pressure |p-| [Pa].
///     ppp_pa: Peak positive pressure p+ [Pa].
///     sigma_arr: Goldberg shock parameters.
///
/// Returns:
///     Effective pressure array [Pa], same length as sigma_arr.
///
/// Reference:
///     Hamilton & Blackstock (1998) Nonlinear Acoustics, S3.3.
#[pyfunction]
#[pyo3(signature = (pnp_pa, ppp_pa, sigma_arr))]
pub fn shock_waveform_pressure(
    py: Python<'_>,
    pnp_pa: f64,
    ppp_pa: f64,
    sigma_arr: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let s_s = sigma_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::shock_waveform_pressure(pnp_pa, ppp_pa, s_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Shock-enhanced volumetric heat-source density.
///
/// Q_eff(sigma) = G(sigma) * alpha * p_eff^2 / (rho * c)
///
/// Args:
///     p_eff_arr: Effective pressure array [Pa].
///     sigma_arr: Goldberg shock parameters (same length as p_eff_arr).
///     alpha_np_m: Linear attenuation [Np/m].
///     rho: Density [kg/m3].
///     c: Sound speed [m/s].
///
/// Returns:
///     Heat-source density array [W/m3], same length as p_eff_arr.
///
/// Reference:
///     Hamilton & Blackstock (1998) Nonlinear Acoustics, S4.3.
#[pyfunction]
#[pyo3(signature = (p_eff_arr, sigma_arr, alpha_np_m, rho, c))]
pub fn shock_heat_source_density(
    py: Python<'_>,
    p_eff_arr: PyReadonlyArray1<f64>,
    sigma_arr: PyReadonlyArray1<f64>,
    alpha_np_m: f64,
    rho: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = p_eff_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let s_s = sigma_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::shock_heat_source_density(p_s, s_s, alpha_np_m, rho, c);
    Ok(result.to_pyarray(py).unbind())
}

/// Rectangular-envelope Fubini waveform for a shock-formed millisecond pulse.
///
/// Args:
///     t_arr: Time sample points [s].
///     p0_pa: Peak pressure amplitude [Pa].
///     f0: Fundamental frequency [Hz].
///     duration_s: Pulse duration [s].
///     t_start: Burst start time [s].
///     sigma: Fubini-Euler parameter (0 <= sigma < 1).
///     n_max: Highest harmonic order.
///
/// Returns:
///     Pressure waveform [Pa], same length as t_arr.
///
/// Reference:
///     Hamilton & Blackstock (1998) Nonlinear Acoustics, S3.3.
///     Khokhlova et al. (2014) Int. J. Hyperthermia 31, 145.
#[pyfunction]
#[pyo3(signature = (t_arr, p0_pa, f0, duration_s, t_start, sigma, n_max))]
pub fn shock_vapor_pulse_waveform(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    p0_pa: f64,
    f0: f64,
    duration_s: f64,
    t_start: f64,
    sigma: f64,
    n_max: u32,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        wave::shock_vapor_pulse_waveform(t_s, p0_pa, f0, duration_s, t_start, sigma, n_max);
    Ok(result.to_pyarray(py).unbind())
}

/// Reconstruct the Fubini time-domain waveform from its harmonic series.
///
/// p(t) = p₀ · Σ_{n=1}^{n_max} Bₙ(σ) · sin(n·ω·t),  ω = 2π·f₀
///
/// Args:
///     t_arr: Time sample points [s].
///     p0_pa: Source pressure amplitude [Pa].
///     freq_hz: Fundamental frequency [Hz].
///     sigma: Fubini–Euler nonlinearity parameter (0 ≤ σ < 1).
///     n_max: Highest harmonic order to include.
///
/// Returns:
///     Pressure waveform [Pa], same length as t_arr.
///
/// Reference:
///     Hamilton & Blackstock (1998) Nonlinear Acoustics, §3.3.
#[pyfunction]
#[pyo3(signature = (t_arr, p0_pa, freq_hz, sigma, n_max))]
pub fn fubini_waveform(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    p0_pa: f64,
    freq_hz: f64,
    sigma: f64,
    n_max: u32,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::fubini_waveform(t_s, p0_pa, freq_hz, sigma, n_max);
    Ok(result.to_pyarray(py).unbind())
}

/// Extract Hann-windowed harmonic amplitudes from row-major time traces.
///
/// Args:
///     traces: 2-D pressure trace array with shape (n_traces, n_samples).
///     dt_s: Sample period [s].
///     fundamental_hz: Fundamental frequency [Hz].
///     n_harmonics: Number of harmonics to extract.
///
/// Returns:
///     ndarray of shape (n_traces, n_harmonics), amplitudes [same units as traces].
#[pyfunction]
#[pyo3(signature = (traces, dt_s, fundamental_hz, n_harmonics))]
pub fn hann_windowed_harmonic_amplitudes(
    py: Python<'_>,
    traces: PyReadonlyArray2<f64>,
    dt_s: f64,
    fundamental_hz: f64,
    n_harmonics: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let trace_view = traces.as_array();
    let (n_traces, n_samples) = trace_view.dim();
    let trace_values: Vec<f64> = trace_view.iter().copied().collect();
    let amplitudes = wave::hann_windowed_harmonic_amplitudes(
        &trace_values,
        n_traces,
        n_samples,
        dt_s,
        fundamental_hz,
        n_harmonics,
    )
    .map_err(PyValueError::new_err)?;
    let arr2d = Array2::from_shape_vec((n_traces, n_harmonics), amplitudes)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.to_pyarray(py).unbind())
}

/// Compute Westervelt harmonic pressure evolution along a propagation axis.
///
/// Returns a 2-D array of shape (len(z_arr), n_max) containing the pressure
/// amplitude [Pa] of each harmonic at each axial position.
///
/// Args:
///     z_arr: Axial positions [m].
///     p0: Source pressure amplitude [Pa].
///     f0: Fundamental frequency [Hz].
///     c0: Sound speed [m/s].
///     rho0: Density [kg/m³].
///     beta: Nonlinearity coefficient.
///     alpha_np_m: Absorption at f0 [Np/m].
///     n_max: Number of harmonics to compute.
///
/// Returns:
///     ndarray of shape (nz, n_max).
#[pyfunction]
#[pyo3(signature = (z_arr, p0, f0, c0, rho0, beta, alpha_np_m, n_max))]
pub fn westervelt_harmonic_evolution(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    p0: f64,
    f0: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
    alpha_np_m: f64,
    n_max: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let z_slice = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rows: Vec<Vec<f64>> =
        wave::westervelt_harmonic_evolution(z_slice, p0, f0, c0, rho0, beta, alpha_np_m, n_max);
    let nz = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr2d = Array2::from_shape_vec((nz, n_max), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.to_pyarray(py).unbind())
}

