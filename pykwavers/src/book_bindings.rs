// book_bindings.rs — PyO3 bindings for kwavers::physics::book::*
//
// Wraps all analytical / textbook-formula functions from the book physics
// sub-modules so they are callable from Python without additional glue code.
//
// Organisation mirrors the sub-module layout:
//   wave, transducer, cavitation, tissue, safety, skull, photoacoustics,
//   elastography, imaging, thermal, inverse, sonogenetics, rtm

use kwavers::physics::book::{
    cavitation, elastography, imaging, inverse as inverse_mod, photoacoustics,
    rtm as rtm_mod, safety, skull as skull_mod, sonogenetics, thermal, tissue, transducer, wave,
};
use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ============================================================================
// wave.rs
// ============================================================================

/// Compute a 1-D standing-wave pressure field.
///
/// p(x, t) = 2 * p0 * cos(k*x) * cos(omega*t)
///
/// Args:
///     p0: Peak amplitude [Pa].
///     k: Wave number [rad/m].
///     x: Spatial positions [m] (1-D array).
///     omega_t: Phase angle omega*t [rad].
///
/// Returns:
///     Pressure array [Pa] of the same length as *x*.
#[pyfunction]
#[pyo3(signature = (p0, k, x, omega_t))]
fn standing_wave_1d(
    py: Python<'_>,
    p0: f64,
    k: f64,
    x: PyReadonlyArray1<f64>,
    omega_t: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::standing_wave_1d(p0, k, x_slice, omega_t);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute a 1-D plane-wave pressure field.
///
/// p(x, t) = amplitude * cos(k*x - omega*t)
///
/// Args:
///     amplitude: Amplitude [Pa].
///     k: Wave number [rad/m].
///     x: Spatial positions [m].
///     omega_t: Phase angle omega*t [rad].
///
/// Returns:
///     Pressure array [Pa].
#[pyfunction]
#[pyo3(signature = (amplitude, k, x, omega_t))]
fn plane_wave_pressure_1d(
    py: Python<'_>,
    amplitude: f64,
    k: f64,
    x: PyReadonlyArray1<f64>,
    omega_t: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_slice = x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::plane_wave_pressure_1d(amplitude, k, x_slice, omega_t);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute spherical-wave pressure at radial distances *r*.
///
/// p(r) = amplitude / r * cos(k*r)
///
/// Args:
///     amplitude: Source strength [Pa·m].
///     k: Wave number [rad/m].
///     r: Radial distances [m].
///
/// Returns:
///     Pressure array [Pa].
#[pyfunction]
#[pyo3(signature = (amplitude, k, r))]
fn spherical_wave_pressure(
    py: Python<'_>,
    amplitude: f64,
    k: f64,
    r: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let r_slice = r
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::spherical_wave_pressure(amplitude, k, r_slice);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the pressure reflection coefficient at a planar interface.
///
/// R = (z2 - z1) / (z2 + z1)
///
/// Args:
///     z1: Acoustic impedance of medium 1 [Pa·s/m].
///     z2: Acoustic impedance of medium 2 [Pa·s/m].
///
/// Returns:
///     Pressure reflection coefficient (dimensionless).
#[pyfunction]
#[pyo3(signature = (z1, z2))]
fn reflection_pressure_coeff(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(wave::reflection_pressure_coeff(z1, z2))
}

/// Compute the pressure transmission coefficient at a planar interface.
///
/// T = 2*z2 / (z2 + z1)
///
/// Args:
///     z1: Acoustic impedance of medium 1 [Pa·s/m].
///     z2: Acoustic impedance of medium 2 [Pa·s/m].
///
/// Returns:
///     Pressure transmission coefficient (dimensionless).
#[pyfunction]
#[pyo3(signature = (z1, z2))]
fn transmission_pressure_coeff(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(wave::transmission_pressure_coeff(z1, z2))
}

/// Compute power-law attenuation α(f) = α0 * f^y in Np/m.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     alpha0: Attenuation coefficient at 1 Hz [Np/m/Hz^y].
///     y: Frequency power-law exponent.
///
/// Returns:
///     Attenuation array [Np/m].
#[pyfunction]
#[pyo3(signature = (f_hz, alpha0, y))]
fn power_law_attenuation_np_m(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    alpha0: f64,
    y: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_slice = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::power_law_attenuation_np_m(f_slice, alpha0, y);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute power-law absorption α(f) = α0 * f^y in dB/(cm·MHz^y).
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     alpha0: Coefficient [dB/(cm·MHz^y)].
///     y: Power-law exponent.
///
/// Returns:
///     Absorption array [dB/cm].
#[pyfunction]
#[pyo3(signature = (f_mhz, alpha0, y))]
fn absorption_power_law_db_cm(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    alpha0: f64,
    y: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_slice = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::absorption_power_law_db_cm(f_slice, alpha0, y);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute FDTD numerical phase error for a 1-D Yee grid.
///
/// Args:
///     kh: k*h (wave-number times grid spacing) array.
///     cfl: Courant–Friedrichs–Lewy number.
///
/// Returns:
///     Relative phase error array.
#[pyfunction]
#[pyo3(signature = (kh, cfl))]
fn fdtd_phase_error_1d(
    py: Python<'_>,
    kh: PyReadonlyArray1<f64>,
    cfl: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::fdtd_phase_error_1d(kh_slice, cfl);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute PSTD numerical phase error.
///
/// Args:
///     kh: k*h array.
///
/// Returns:
///     Relative phase error array.
#[pyfunction]
#[pyo3(signature = (kh,))]
fn pstd_phase_error(
    py: Python<'_>,
    kh: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::pstd_phase_error(kh_slice);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute k-space dispersion-correction phase error.
///
/// Args:
///     kh: k*h array.
///     cfl: CFL number.
///
/// Returns:
///     Residual phase error array after k-space correction.
#[pyfunction]
#[pyo3(signature = (kh, cfl))]
fn kspace_correction_error(
    py: Python<'_>,
    kh: PyReadonlyArray1<f64>,
    cfl: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::kspace_correction_error(kh_slice, cfl);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the FDTD CFL stability limit for an n-dimensional grid.
///
/// CFL_max = 1 / sqrt(ndim)
///
/// Args:
///     ndim: Number of spatial dimensions (1, 2, or 3).
///
/// Returns:
///     Maximum stable CFL number.
#[pyfunction]
#[pyo3(signature = (ndim,))]
fn fdtd_cfl_limit(ndim: u32) -> PyResult<f64> {
    Ok(wave::fdtd_cfl_limit(ndim))
}

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
fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> PyResult<f64> {
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
fn fubini_harmonic_spectrum(
    py: Python<'_>,
    n_max: u32,
    sigma: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = wave::fubini_harmonic_spectrum(n_max, sigma);
    Ok(result.into_pyarray(py).unbind())
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
fn shock_formation_distance(
    p0_pa: f64,
    f0_hz: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
) -> PyResult<f64> {
    Ok(wave::shock_formation_distance(p0_pa, f0_hz, c0, rho0, beta))
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
fn westervelt_harmonic_evolution(
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
    Ok(arr2d.into_pyarray(py).unbind())
}

// ============================================================================
// transducer.rs
// ============================================================================

/// Compute the circular-piston far-field directivity pattern.
///
/// D(theta) = 2 * J1(ka*sin(theta)) / (ka*sin(theta))
///
/// Args:
///     theta_rad: Observation angles [rad].
///     ka: Wave-number × radius product.
///
/// Returns:
///     Directivity array (normalised to unity on-axis).
#[pyfunction]
#[pyo3(signature = (theta_rad, ka))]
fn circular_piston_directivity(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    ka: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::circular_piston_directivity(t_slice, ka);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the linear-array factor as a function of angle.
///
/// Args:
///     theta_rad: Observation angles [rad].
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     n: Number of elements.
///     steer_rad: Electronic steering angle [rad].
///
/// Returns:
///     Normalised array factor.
#[pyfunction]
#[pyo3(signature = (theta_rad, k, d_m, n, steer_rad))]
fn linear_array_factor(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    k: f64,
    d_m: f64,
    n: usize,
    steer_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::linear_array_factor(t_slice, k, d_m, n, steer_rad);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute grating-lobe angles for a uniform linear array.
///
/// Args:
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     steer_rad: Steering angle [rad].
///
/// Returns:
///     Array of grating-lobe angles [rad] (may be empty if none exist).
#[pyfunction]
#[pyo3(signature = (k, d_m, steer_rad))]
fn grating_lobe_angles(
    py: Python<'_>,
    k: f64,
    d_m: f64,
    steer_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = transducer::grating_lobe_angles(k, d_m, steer_rad);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute element apodization weights for a given window type.
///
/// Supported window types: "uniform", "hann", "hamming", "blackman",
/// "nuttall", "tukey".
///
/// Args:
///     n: Number of elements.
///     window_type: Name of the window function.
///
/// Returns:
///     Weight array of length *n*.
#[pyfunction]
#[pyo3(signature = (n, window_type))]
fn apodization_weights(
    py: Python<'_>,
    n: usize,
    window_type: String,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = transducer::apodization_weights(n, &window_type);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute time-delay laws for 2-D geometric focusing.
///
/// Args:
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     x_f: Focal point x-coordinate [m].
///     z_f: Focal point z-coordinate [m].
///     c: Sound speed [m/s].
///
/// Returns:
///     Delay array [s], same length as *elem_x*.
#[pyfunction]
#[pyo3(signature = (elem_x, elem_z, x_f, z_f, c))]
fn delay_law_focus_2d(
    py: Python<'_>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::delay_law_focus_2d(ex, ez, x_f, z_f, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the complex 2-D beam pattern for a phased array.
///
/// Returns a list [real_field, imag_field], each a 2-D ndarray of shape
/// (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///     weights: Apodization weights per element.
///     delays: Delay per element [s].
///
/// Returns:
///     [real_nx_nz, imag_nx_nz] — list of two 2-D arrays.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, elem_x, elem_z, freq_hz, c, weights, delays))]
fn beam_pattern_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
    weights: PyReadonlyArray1<f64>,
    delays: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let w_s = weights
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d_s = delays
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let (real_flat, imag_flat) =
        transducer::beam_pattern_2d(x_s, z_s, ex, ez, freq_hz, c, w_s, d_s);
    let real_arr = Array2::from_shape_vec((nx, nz), real_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let imag_arr = Array2::from_shape_vec((nx, nz), imag_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        real_arr.into_pyarray(py).unbind(),
        imag_arr.into_pyarray(py).unbind(),
    ))
}

/// Compute the on-axis pressure of a circular piston transducer.
///
/// Args:
///     z_arr: Axial positions [m].
///     radius_m: Piston radius [m].
///     freq_hz: Frequency [Hz].
///     p0_pa: Surface pressure amplitude [Pa].
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude [Pa].
#[pyfunction]
#[pyo3(signature = (z_arr, radius_m, freq_hz, p0_pa, c))]
fn circular_piston_onaxis(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    radius_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::circular_piston_onaxis(z_s, radius_m, freq_hz, p0_pa, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the on-axis pressure of a focused-bowl (spherically focused) transducer.
///
/// Args:
///     z_arr: Axial positions [m].
///     bowl_radius_m: Bowl aperture radius [m].
///     focal_length_m: Geometric focal length [m].
///     freq_hz: Frequency [Hz].
///     p0_pa: Source pressure [Pa].
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude [Pa].
#[pyfunction]
#[pyo3(signature = (z_arr, bowl_radius_m, focal_length_m, freq_hz, p0_pa, c))]
fn focused_bowl_onaxis(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    bowl_radius_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        transducer::focused_bowl_onaxis(z_s, bowl_radius_m, focal_length_m, freq_hz, p0_pa, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute band-limited interpolation (BLI) stencil weights.
///
/// Args:
///     delta: Sub-sample offsets (1-D array).
///     n_stencil: Number of stencil points.
///
/// Returns:
///     2-D array of shape (len(delta), n_stencil).
#[pyfunction]
#[pyo3(signature = (delta, n_stencil))]
fn bli_stencil_weights(
    py: Python<'_>,
    delta: PyReadonlyArray1<f64>,
    n_stencil: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let d_s = delta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rows: Vec<Vec<f64>> = transducer::bli_stencil_weights(d_s, n_stencil);
    let n_delta = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr2d = Array2::from_shape_vec((n_delta, n_stencil), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

// ============================================================================
// cavitation.rs
// ============================================================================

/// Compute the Minnaert resonance frequency of a free bubble.
///
/// f_r = (1/(2*pi*r0)) * sqrt(3*gamma*p0 / rho)
///
/// Args:
///     r0_m: Equilibrium bubble radius [m].
///     gamma: Polytropic index.
///     p0_pa: Ambient pressure [Pa].
///     rho: Liquid density [kg/m³].
///
/// Returns:
///     Resonance frequency [Hz].
#[pyfunction]
#[pyo3(signature = (r0_m, gamma, p0_pa, rho))]
fn minnaert_resonance_hz(r0_m: f64, gamma: f64, p0_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(cavitation::minnaert_resonance_hz(r0_m, gamma, p0_pa, rho))
}

/// Compute the Blake cavitation threshold pressure.
///
/// Args:
///     r0_m: Initial bubble radius [m].
///     p0_pa: Ambient pressure [Pa].
///     sigma_n_m: Surface tension [N/m].
///
/// Returns:
///     Blake threshold negative pressure [Pa].
#[pyfunction]
#[pyo3(signature = (r0_m, p0_pa, sigma_n_m))]
fn blake_threshold_pa(r0_m: f64, p0_pa: f64, sigma_n_m: f64) -> PyResult<f64> {
    Ok(cavitation::blake_threshold_pa(r0_m, p0_pa, sigma_n_m))
}

/// Compute the Rayleigh collapse time of an empty spherical cavity.
///
/// t_c = 0.9147 * r_max * sqrt(rho / p_inf)
///
/// Args:
///     rmax_m: Maximum bubble radius [m].
///     p_inf_pa: Ambient pressure [Pa].
///     rho: Liquid density [kg/m³].
///
/// Returns:
///     Collapse time [s].
#[pyfunction]
#[pyo3(signature = (rmax_m, p_inf_pa, rho))]
fn rayleigh_collapse_time_s(rmax_m: f64, p_inf_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(cavitation::rayleigh_collapse_time_s(rmax_m, p_inf_pa, rho))
}

/// Integrate the Rayleigh–Plesset equation with RK4.
///
/// Args:
///     r0_m: Initial radius [m].
///     rdot0: Initial wall velocity [m/s].
///     p_ac_pa: Acoustic pressure amplitude [Pa].
///     freq_hz: Driving frequency [Hz].
///     t_arr: Time array [s].
///     p0_pa: Ambient pressure [Pa].
///     rho: Liquid density [kg/m³].
///     sigma: Surface tension [N/m].
///     mu: Dynamic viscosity [Pa·s].
///     kappa: Polytropic index.
///     p_v_pa: Vapour pressure [Pa].
///
/// Returns:
///     (r, rdot) — tuple of radius [m] and wall-velocity [m/s] arrays.
#[pyfunction]
#[pyo3(signature = (r0_m, rdot0, p_ac_pa, freq_hz, t_arr, p0_pa, rho, sigma, mu, kappa, p_v_pa))]
fn rayleigh_plesset_rk4(
    py: Python<'_>,
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: PyReadonlyArray1<f64>,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (r, rdot) = cavitation::rayleigh_plesset_rk4(
        r0_m, rdot0, p_ac_pa, freq_hz, t_s, p0_pa, rho, sigma, mu, kappa, p_v_pa,
    );
    Ok((
        r.into_pyarray(py).unbind(),
        rdot.into_pyarray(py).unbind(),
    ))
}

/// Integrate the Keller–Miksis equation with RK4.
///
/// Extends Rayleigh–Plesset to include liquid compressibility via *c_liquid*.
///
/// Args:
///     r0_m: Initial radius [m].
///     rdot0: Initial wall velocity [m/s].
///     p_ac_pa: Acoustic driving amplitude [Pa].
///     freq_hz: Frequency [Hz].
///     t_arr: Time array [s].
///     p0_pa: Ambient pressure [Pa].
///     rho: Density [kg/m³].
///     sigma: Surface tension [N/m].
///     mu: Viscosity [Pa·s].
///     kappa: Polytropic index.
///     p_v_pa: Vapour pressure [Pa].
///     c_liquid: Sound speed in the liquid [m/s].
///
/// Returns:
///     (r, rdot) tuple.
#[pyfunction]
#[pyo3(signature = (r0_m, rdot0, p_ac_pa, freq_hz, t_arr, p0_pa, rho, sigma, mu, kappa, p_v_pa, c_liquid))]
fn keller_miksis_rk4(
    py: Python<'_>,
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: PyReadonlyArray1<f64>,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (r, rdot) = cavitation::keller_miksis_rk4(
        r0_m, rdot0, p_ac_pa, freq_hz, t_s, p0_pa, rho, sigma, mu, kappa, p_v_pa, c_liquid,
    );
    Ok((
        r.into_pyarray(py).unbind(),
        rdot.into_pyarray(py).unbind(),
    ))
}

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
fn bubble_power_spectrum(
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

// ============================================================================
// tissue.rs
// ============================================================================

/// Compute the sound speed of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
fn water_sound_speed_temperature(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::water_sound_speed_temperature(t_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the density of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Density array [kg/m³].
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
fn water_density_temperature(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::water_density_temperature(t_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Return the B/A nonlinearity parameter for a named medium.
///
/// Supported values: "water", "blood", "fat", "liver", "kidney", "brain",
/// "muscle", "bone".
///
/// Args:
///     medium: Medium name string.
///
/// Returns:
///     B/A value.
#[pyfunction]
#[pyo3(signature = (medium,))]
fn ba_parameter(medium: String) -> PyResult<f64> {
    Ok(tissue::ba_parameter(&medium))
}

/// Compute frequency-dependent tissue absorption in dB/cm.
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     tissue: Tissue name string.
///
/// Returns:
///     Absorption array [dB/cm].
#[pyfunction]
#[pyo3(signature = (f_mhz, tissue))]
fn tissue_absorption_db_cm(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    tissue: String,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::tissue_absorption_db_cm(f_s, &tissue);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Kramers–Kronig consistent sound speed dispersion.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     alpha0: Attenuation coefficient [Np/m/Hz^y].
///     y: Power-law exponent.
///     f_ref_hz: Reference frequency [Hz].
///     c_ref: Sound speed at *f_ref_hz* [m/s].
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (f_hz, alpha0, y, f_ref_hz, c_ref))]
fn kramers_kronig_sound_speed(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    alpha0: f64,
    y: f64,
    f_ref_hz: f64,
    c_ref: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::kramers_kronig_sound_speed(f_s, alpha0, y, f_ref_hz, c_ref);
    Ok(result.into_pyarray(py).unbind())
}

/// Return tabulated acoustic properties for a named tissue.
///
/// Args:
///     tissue: Tissue name string.
///
/// Returns:
///     (sound_speed_m_s, density_kg_m3, attenuation_db_cm_mhz,
///      nonlinearity_ba, impedance_mrayl) — all f64.
#[pyfunction]
#[pyo3(signature = (tissue,))]
fn tissue_properties(tissue: String) -> PyResult<(f64, f64, f64, f64, f64)> {
    Ok(tissue::tissue_properties(&tissue))
}

// ============================================================================
// safety.rs
// ============================================================================

/// Compute the Mechanical Index (MI).
///
/// MI = |p_neg_pa| / (1e6 * sqrt(f_hz / 1e6))
///
/// Args:
///     p_neg_pa: Peak negative pressure [Pa].
///     f_hz: Frequency [Hz].
///
/// Returns:
///     Mechanical Index (dimensionless).
#[pyfunction]
#[pyo3(signature = (p_neg_pa, f_hz))]
fn mechanical_index(p_neg_pa: f64, f_hz: f64) -> PyResult<f64> {
    Ok(safety::mechanical_index(p_neg_pa, f_hz))
}

/// Compute the Thermal Index for soft tissue (TIS).
///
/// Args:
///     wstp_mw: W_STP — time-averaged power at the surface [mW].
///     f_mhz: Frequency [MHz].
///
/// Returns:
///     TIS value.
#[pyfunction]
#[pyo3(signature = (wstp_mw, f_mhz))]
fn thermal_index_soft_tissue(wstp_mw: f64, f_mhz: f64) -> PyResult<f64> {
    Ok(safety::thermal_index_soft_tissue(wstp_mw, f_mhz))
}

/// Compute the Thermal Index for bone (TIB).
///
/// Args:
///     w_mw: Beam power at bone surface [mW].
///     f_mhz: Frequency [MHz].
///
/// Returns:
///     TIB value.
#[pyfunction]
#[pyo3(signature = (w_mw, f_mhz))]
fn thermal_index_bone(w_mw: f64, f_mhz: f64) -> PyResult<f64> {
    Ok(safety::thermal_index_bone(w_mw, f_mhz))
}

/// Compute the cumulative CEM43 thermal dose over a temperature time series.
///
/// Args:
///     t_celsius: Temperature time series [°C].
///     dt_s: Time-step [s].
///
/// Returns:
///     Cumulative CEM43 array [min].
#[pyfunction]
#[pyo3(signature = (t_celsius, dt_s))]
fn cem43_cumulative(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
    dt_s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::cem43_cumulative(t_s, dt_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Arrhenius thermal-damage integral Ω.
///
/// Ω = A * ∫ exp(-Ea / (R*T(t))) dt
///
/// Args:
///     t_celsius: Temperature time series [°C].
///     dt_s: Time-step [s].
///     a_per_s: Pre-exponential frequency factor [1/s].
///     ea_j_mol: Activation energy [J/mol].
///
/// Returns:
///     Damage integral Ω (dimensionless).
#[pyfunction]
#[pyo3(signature = (t_celsius, dt_s, a_per_s, ea_j_mol))]
fn arrhenius_damage_integral(
    t_celsius: PyReadonlyArray1<f64>,
    dt_s: f64,
    a_per_s: f64,
    ea_j_mol: f64,
) -> PyResult<f64> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(safety::arrhenius_damage_integral(t_s, dt_s, a_per_s, ea_j_mol))
}

/// Return the FDA ISPTA diagnostic-ultrasound limit (720 mW/cm²).
///
/// Returns:
///     ISPTA limit [mW/cm²].
#[pyfunction]
fn fda_ispta_limit_mw_cm2() -> PyResult<f64> {
    Ok(safety::fda_ispta_limit_mw_cm2())
}

/// Return the FDA ISPPA diagnostic-ultrasound limit (190 W/cm²).
///
/// Returns:
///     ISPPA limit [W/cm²].
#[pyfunction]
fn fda_isppa_limit_w_cm2() -> PyResult<f64> {
    Ok(safety::fda_isppa_limit_w_cm2())
}

// ============================================================================
// skull.rs
// ============================================================================

/// Compute two-way skull insertion loss using a power-law attenuation model.
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     thickness_cm: Skull thickness [cm].
///     alpha0: Attenuation coefficient [dB/(cm·MHz)].
///
/// Returns:
///     Two-way insertion loss array [dB].
#[pyfunction]
#[pyo3(signature = (f_mhz, thickness_cm, alpha0))]
fn skull_insertion_loss_two_way_db(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    thickness_cm: f64,
    alpha0: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::skull_insertion_loss_two_way_db(f_s, thickness_cm, alpha0);
    Ok(result.into_pyarray(py).unbind())
}

/// Generate a random phase screen modelling skull aberration.
///
/// Args:
///     n: Number of phase-screen points.
///     sigma_phi_rad: Phase standard deviation [rad].
///     seed: RNG seed for reproducibility.
///
/// Returns:
///     Phase array [rad] of length *n*.
#[pyfunction]
#[pyo3(signature = (n, sigma_phi_rad, seed))]
fn skull_phase_screen(
    py: Python<'_>,
    n: usize,
    sigma_phi_rad: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = skull_mod::skull_phase_screen(n, sigma_phi_rad, seed);
    Ok(result.into_pyarray(py).unbind())
}

/// Convert Hounsfield units to sound speed using the Schneider model.
///
/// Args:
///     hu: HU array.
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (hu,))]
fn hu_to_sound_speed_schneider(
    py: Python<'_>,
    hu: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let h_s = hu
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::hu_to_sound_speed_schneider(h_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Convert Hounsfield units to density using the Schneider model.
///
/// Args:
///     hu: HU array.
///
/// Returns:
///     Density array [kg/m³].
#[pyfunction]
#[pyo3(signature = (hu,))]
fn hu_to_density_schneider(
    py: Python<'_>,
    hu: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let h_s = hu
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = skull_mod::hu_to_density_schneider(h_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Strehl ratio for a given wavefront-error standard deviation.
///
/// S ≈ exp(-sigma_phi²)  (Maréchal approximation)
///
/// Args:
///     sigma_phi_rad: RMS wavefront error [rad].
///
/// Returns:
///     Strehl ratio (0–1).
#[pyfunction]
#[pyo3(signature = (sigma_phi_rad,))]
fn strehl_ratio(sigma_phi_rad: f64) -> PyResult<f64> {
    Ok(skull_mod::strehl_ratio(sigma_phi_rad))
}

/// Compute skull surface temperature rise due to a heat-flux boundary.
///
/// Args:
///     t_arr: Time array [s].
///     heat_flux: Applied heat flux [W/m²].
///     k_skull: Skull thermal conductivity [W/(m·K)].
///     rho_skull: Skull density [kg/m³].
///     cp_skull: Skull specific heat capacity [J/(kg·K)].
///
/// Returns:
///     Surface temperature-rise array [°C].
#[pyfunction]
#[pyo3(signature = (t_arr, heat_flux, k_skull, rho_skull, cp_skull))]
fn skull_surface_temperature_rise(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    heat_flux: f64,
    k_skull: f64,
    rho_skull: f64,
    cp_skull: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        skull_mod::skull_surface_temperature_rise(t_s, heat_flux, k_skull, rho_skull, cp_skull);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the complex transmission coefficient through a skull layer
/// using the transfer-matrix method.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     z_water: Water impedance [Pa·s/m].
///     z_skull: Skull impedance [Pa·s/m].
///     z_brain: Brain impedance [Pa·s/m].
///     c_skull: Skull sound speed [m/s].
///     d_skull_m: Skull thickness [m].
///
/// Returns:
///     Python complex number T = |T| * exp(i*phi).
#[pyfunction]
#[pyo3(signature = (f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m))]
fn skull_transfer_matrix_transmission(
    py: Python<'_>,
    f_hz: f64,
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> PyResult<PyObject> {
    let c: Complex64 = skull_mod::skull_transfer_matrix_transmission(
        f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m,
    );
    let builtins = py.import("builtins")?;
    let py_complex = builtins.getattr("complex")?;
    Ok(py_complex.call1((c.re, c.im))?.into())
}

/// Compute the skull transmission spectrum (magnitude and phase).
///
/// Args:
///     f_hz: Frequency array [Hz].
///     z_water: Water impedance [Pa·s/m].
///     z_skull: Skull impedance [Pa·s/m].
///     z_brain: Brain impedance [Pa·s/m].
///     c_skull: Skull sound speed [m/s].
///     d_skull_m: Skull thickness [m].
///
/// Returns:
///     (magnitude_array, phase_array_rad) tuple.
#[pyfunction]
#[pyo3(signature = (f_hz, z_water, z_skull, z_brain, c_skull, d_skull_m))]
fn skull_transmission_spectrum(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (mag, phase) = skull_mod::skull_transmission_spectrum(
        f_s, z_water, z_skull, z_brain, c_skull, d_skull_m,
    );
    Ok((
        mag.into_pyarray(py).unbind(),
        phase.into_pyarray(py).unbind(),
    ))
}

// ============================================================================
// photoacoustics.rs
// ============================================================================

/// Return the molar absorption spectrum of oxyhaemoglobin (HbO2).
///
/// Args:
///     wavelength_nm: Wavelength array [nm].
///
/// Returns:
///     Molar absorption coefficient array [cm⁻¹/M].
#[pyfunction]
#[pyo3(signature = (wavelength_nm,))]
fn hbo2_molar_absorption(
    py: Python<'_>,
    wavelength_nm: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let w_s = wavelength_nm
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::hbo2_molar_absorption(w_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Return the molar absorption spectrum of deoxyhaemoglobin (Hb).
///
/// Args:
///     wavelength_nm: Wavelength array [nm].
///
/// Returns:
///     Molar absorption coefficient array [cm⁻¹/M].
#[pyfunction]
#[pyo3(signature = (wavelength_nm,))]
fn hb_molar_absorption(
    py: Python<'_>,
    wavelength_nm: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let w_s = wavelength_nm
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::hb_molar_absorption(w_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Grüneisen parameter of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Grüneisen parameter array (dimensionless).
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
fn gruneisen_parameter_water(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::gruneisen_parameter_water(t_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the photoacoustic pressure signal from an absorbing sphere.
///
/// Args:
///     t_arr: Time array [s].
///     r0_m: Sphere radius [m].
///     gamma: Grüneisen parameter.
///     mua_per_m: Absorption coefficient [1/m].
///     c: Sound speed [m/s].
///     r_det_m: Detector distance [m].
///     initial_pressure_pa: Initial pressure rise [Pa].
///
/// Returns:
///     Pressure signal array [Pa].
#[pyfunction]
#[pyo3(signature = (t_arr, r0_m, gamma, mua_per_m, c, r_det_m, initial_pressure_pa))]
fn pa_sphere_pressure_signal(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    r0_m: f64,
    gamma: f64,
    mua_per_m: f64,
    c: f64,
    r_det_m: f64,
    initial_pressure_pa: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::pa_sphere_pressure_signal(
        t_s,
        r0_m,
        gamma,
        mua_per_m,
        c,
        r_det_m,
        initial_pressure_pa,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the photoacoustic axial resolution.
///
/// δz ≈ 0.88 * c / bandwidth
///
/// Args:
///     bandwidth_hz: Transducer bandwidth [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Axial resolution [m].
#[pyfunction]
#[pyo3(signature = (bandwidth_hz, c))]
fn pa_axial_resolution(bandwidth_hz: f64, c: f64) -> PyResult<f64> {
    Ok(photoacoustics::pa_axial_resolution(bandwidth_hz, c))
}

/// Estimate chromophore concentrations by least-squares spectral unmixing.
///
/// Solves: spectra_matrix @ concentrations ≈ measurements
///
/// Args:
///     spectra_matrix: Absorption spectra matrix (n_wavelengths × n_chromophores).
///     measurements: Measured PA signals (length n_wavelengths).
///
/// Returns:
///     Concentration vector (length n_chromophores).
#[pyfunction]
#[pyo3(signature = (spectra_matrix, measurements))]
fn spectroscopic_unmixing_lstsq(
    py: Python<'_>,
    spectra_matrix: PyReadonlyArray2<f64>,
    measurements: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let sm = spectra_matrix.as_array();
    let (nrows, _ncols) = sm.dim();
    let spectra_vecs: Vec<Vec<f64>> = (0..nrows).map(|i| sm.row(i).to_vec()).collect();
    let meas_slice = measurements
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::spectroscopic_unmixing_lstsq(&spectra_vecs, meas_slice);
    Ok(result.into_pyarray(py).unbind())
}

// ============================================================================
// elastography.rs
// ============================================================================

/// Compute the shear-wave speed from shear modulus and density.
///
/// c_s = sqrt(mu / rho)
///
/// Args:
///     mu_pa: Shear modulus [Pa].
///     rho: Density [kg/m³].
///
/// Returns:
///     Shear-wave speed [m/s].
#[pyfunction]
#[pyo3(signature = (mu_pa, rho))]
fn shear_wave_speed(mu_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(elastography::shear_wave_speed(mu_pa, rho))
}

/// Compute the Voigt complex shear modulus G*(ω).
///
/// G*(ω) = mu + i*omega*eta
///
/// Args:
///     omega_arr: Angular frequency array [rad/s].
///     mu_pa: Elastic modulus [Pa].
///     eta_pa_s: Viscosity [Pa·s].
///
/// Returns:
///     (real_part, imag_part) tuple of arrays [Pa].
#[pyfunction]
#[pyo3(signature = (omega_arr, mu_pa, eta_pa_s))]
fn voigt_complex_modulus(
    py: Python<'_>,
    omega_arr: PyReadonlyArray1<f64>,
    mu_pa: f64,
    eta_pa_s: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let om_s = omega_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = elastography::voigt_complex_modulus(om_s, mu_pa, eta_pa_s);
    let real: Vec<f64> = result.iter().map(|c| c.re).collect();
    let imag: Vec<f64> = result.iter().map(|c| c.im).collect();
    Ok((
        real.into_pyarray(py).unbind(),
        imag.into_pyarray(py).unbind(),
    ))
}

/// Compute the springpot (fractional Kelvin) complex shear modulus.
///
/// G*(ω) = G0 * (i*ω)^alpha
///
/// Args:
///     omega_arr: Angular frequency array [rad/s].
///     g0: Quasi-static modulus scale [Pa·s^alpha].
///     alpha_exp: Fractional exponent in [0, 1].
///
/// Returns:
///     (real_part, imag_part) tuple of arrays [Pa].
#[pyfunction]
#[pyo3(signature = (omega_arr, g0, alpha_exp))]
fn springpot_complex_modulus(
    py: Python<'_>,
    omega_arr: PyReadonlyArray1<f64>,
    g0: f64,
    alpha_exp: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let om_s = omega_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = elastography::springpot_complex_modulus(om_s, g0, alpha_exp);
    let real: Vec<f64> = result.iter().map(|c| c.re).collect();
    let imag: Vec<f64> = result.iter().map(|c| c.im).collect();
    Ok((
        real.into_pyarray(py).unbind(),
        imag.into_pyarray(py).unbind(),
    ))
}

/// Compute the Voigt shear-wave phase velocity dispersion curve.
///
/// Args:
///     f_arr: Frequency array [Hz].
///     mu_pa: Shear modulus [Pa].
///     eta_pa_s: Viscosity [Pa·s].
///     rho: Density [kg/m³].
///
/// Returns:
///     Phase velocity array [m/s].
#[pyfunction]
#[pyo3(signature = (f_arr, mu_pa, eta_pa_s, rho))]
fn voigt_shear_wave_dispersion(
    py: Python<'_>,
    f_arr: PyReadonlyArray1<f64>,
    mu_pa: f64,
    eta_pa_s: f64,
    rho: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = elastography::voigt_shear_wave_dispersion(f_s, mu_pa, eta_pa_s, rho);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the 2-D MRE displacement field for a harmonic shear wave.
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     shear_speed: Shear-wave phase velocity [m/s].
///     freq_hz: Excitation frequency [Hz].
///     amplitude: Displacement amplitude [m].
///     penetration_depth_m: Exponential decay length [m].
///
/// Returns:
///     Displacement field ndarray of shape (len(x_arr), len(z_arr)) [m].
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, shear_speed, freq_hz, amplitude, penetration_depth_m))]
fn mre_displacement_field(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    shear_speed: f64,
    freq_hz: f64,
    amplitude: f64,
    penetration_depth_m: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let flat = elastography::mre_displacement_field(
        x_s,
        z_s,
        shear_speed,
        freq_hz,
        amplitude,
        penetration_depth_m,
    );
    let arr2d = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

// ============================================================================
// imaging.rs
// ============================================================================

/// Compute the lateral point-spread function using a sinc² model.
///
/// Args:
///     x_arr: Lateral positions [m].
///     f_number: F-number (focal length / aperture).
///     wavelength_m: Acoustic wavelength [m].
///
/// Returns:
///     Normalised lateral PSF array.
#[pyfunction]
#[pyo3(signature = (x_arr, f_number, wavelength_m))]
fn lateral_psf_sinc2(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    f_number: f64,
    wavelength_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = imaging::lateral_psf_sinc2(x_s, f_number, wavelength_m);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the axial point-spread function using a rectangular-spectrum model.
///
/// Args:
///     z_arr: Axial positions [m].
///     c: Sound speed [m/s].
///     bandwidth_hz: Transducer −6 dB bandwidth [Hz].
///
/// Returns:
///     Normalised axial PSF array.
#[pyfunction]
#[pyo3(signature = (z_arr, c, bandwidth_hz))]
fn axial_psf_rect(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    c: f64,
    bandwidth_hz: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = imaging::axial_psf_rect(z_s, c, bandwidth_hz);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Doppler frequency shift.
///
/// f_d = 2 * f0 * v * cos(theta) / c
///
/// Args:
///     v_m_s: Scatterer velocity [m/s].
///     theta_rad: Angle between beam and velocity vector [rad].
///     f0_hz: Transmit centre frequency [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Doppler shift [Hz].
#[pyfunction]
#[pyo3(signature = (v_m_s, theta_rad, f0_hz, c))]
fn doppler_frequency_shift(v_m_s: f64, theta_rad: f64, f0_hz: f64, c: f64) -> PyResult<f64> {
    Ok(imaging::doppler_frequency_shift(v_m_s, theta_rad, f0_hz, c))
}

/// Compute the plane-wave compounding lateral PSF.
///
/// Args:
///     x_arr: Lateral positions [m].
///     n_angles: Number of compounding angles.
///     f_number: F-number.
///     wavelength_m: Wavelength [m].
///
/// Returns:
///     Normalised compounded lateral PSF array.
#[pyfunction]
#[pyo3(signature = (x_arr, n_angles, f_number, wavelength_m))]
fn pw_compounding_lateral_psf(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    n_angles: usize,
    f_number: f64,
    wavelength_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = imaging::pw_compounding_lateral_psf(x_s, n_angles, f_number, wavelength_m);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the −6 dB lateral resolution.
///
/// δx ≈ f_number * wavelength
///
/// Args:
///     f_number: F-number.
///     wavelength_m: Wavelength [m].
///
/// Returns:
///     Lateral resolution [m].
#[pyfunction]
#[pyo3(signature = (f_number, wavelength_m))]
fn lateral_resolution_m(f_number: f64, wavelength_m: f64) -> PyResult<f64> {
    Ok(imaging::lateral_resolution_m(f_number, wavelength_m))
}

// ============================================================================
// thermal.rs
// ============================================================================

/// Simulate focal temperature rise using the Pennes bioheat model.
///
/// Args:
///     t_arr: Time array [s].
///     acoustic_power_w: Absorbed acoustic power [W].
///     focal_volume_m3: Focal volume [m³].
///     k_tissue: Tissue thermal conductivity [W/(m·K)].
///     rho_tissue: Tissue density [kg/m³].
///     cp_tissue: Tissue specific heat [J/(kg·K)].
///     wb_perfusion: Blood perfusion rate [kg/(m³·s)].
///     rho_blood: Blood density [kg/m³].
///     cb_blood: Blood specific heat [J/(kg·K)].
///     t_body_c: Body temperature [°C].
///
/// Returns:
///     Temperature array [°C].
#[pyfunction]
#[pyo3(signature = (t_arr, acoustic_power_w, focal_volume_m3, k_tissue, rho_tissue, cp_tissue, wb_perfusion, rho_blood, cb_blood, t_body_c))]
fn bioheat_focal_temperature_rise(
    py: Python<'_>,
    t_arr: PyReadonlyArray1<f64>,
    acoustic_power_w: f64,
    focal_volume_m3: f64,
    k_tissue: f64,
    rho_tissue: f64,
    cp_tissue: f64,
    wb_perfusion: f64,
    rho_blood: f64,
    cb_blood: f64,
    t_body_c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = thermal::bioheat_focal_temperature_rise(
        t_s,
        acoustic_power_w,
        focal_volume_m3,
        k_tissue,
        rho_tissue,
        cp_tissue,
        wb_perfusion,
        rho_blood,
        cb_blood,
        t_body_c,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the HIFU focal pressure gain (ratio of focal to source pressure).
///
/// Args:
///     aperture_m: Transducer aperture diameter [m].
///     f_number: F-number.
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Focal pressure gain (dimensionless).
#[pyfunction]
#[pyo3(signature = (aperture_m, f_number, freq_hz, c))]
fn hifu_focal_pressure_gain(aperture_m: f64, f_number: f64, freq_hz: f64, c: f64) -> PyResult<f64> {
    Ok(thermal::hifu_focal_pressure_gain(aperture_m, f_number, freq_hz, c))
}

/// Compute the 2-D Gaussian power-deposition distribution.
///
/// Returns a 2-D array of shape (len(r_arr), len(z_arr)) [W/m³].
///
/// Args:
///     r_arr: Radial positions [m].
///     z_arr: Axial positions [m].
///     freq_hz: Frequency [Hz].
///     z_focus_m: Focal depth [m].
///     p0_pa: Source pressure [Pa].
///     c: Sound speed [m/s].
///     rho: Density [kg/m³].
///     alpha_np_m: Attenuation [Np/m].
///     w0_m: Beam waist at focus [m].
///
/// Returns:
///     Power deposition ndarray [W/m³] of shape (nr, nz).
#[pyfunction]
#[pyo3(signature = (r_arr, z_arr, freq_hz, z_focus_m, p0_pa, c, rho, alpha_np_m, w0_m))]
fn gaussian_power_deposition_2d(
    py: Python<'_>,
    r_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    freq_hz: f64,
    z_focus_m: f64,
    p0_pa: f64,
    c: f64,
    rho: f64,
    alpha_np_m: f64,
    w0_m: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let r_s = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nr = r_s.len();
    let nz = z_s.len();
    let flat = thermal::gaussian_power_deposition_2d(
        r_s, z_s, freq_hz, z_focus_m, p0_pa, c, rho, alpha_np_m, w0_m,
    );
    let arr2d = Array2::from_shape_vec((nr, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

// ============================================================================
// inverse.rs
// ============================================================================

/// Build the 1-D Helmholtz finite-difference matrix (Dirichlet BCs).
///
/// Returns an (n × n) sparse-dense matrix A such that (A + k²I)u = f.
///
/// Args:
///     n: Grid points.
///     k: Wave number [rad/m].
///     dx: Grid spacing [m].
///
/// Returns:
///     Dense ndarray of shape (n, n).
#[pyfunction]
#[pyo3(signature = (n, k, dx))]
fn helmholtz_1d_fd_matrix(
    py: Python<'_>,
    n: usize,
    k: f64,
    dx: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let flat = inverse_mod::helmholtz_1d_fd_matrix(n, k, dx);
    let arr2d = Array2::from_shape_vec((n, n), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Compute the singular values of a dense matrix.
///
/// Args:
///     matrix: Dense matrix (nrows × ncols).
///     nrows: Number of rows.
///     ncols: Number of columns.
///
/// Returns:
///     Singular values in descending order.
#[pyfunction]
#[pyo3(signature = (matrix, nrows, ncols))]
fn matrix_singular_values(
    py: Python<'_>,
    matrix: PyReadonlyArray2<f64>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let m = matrix.as_array();
    let flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| m[[i, j]]))
        .collect();
    let result = inverse_mod::matrix_singular_values(&flat, nrows, ncols);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the L-curve (residual norm vs. solution norm) for Tikhonov regularisation.
///
/// Args:
///     a: System matrix (nrows × ncols).
///     b: Right-hand-side vector (length nrows).
///     nrows: Number of rows.
///     ncols: Number of columns.
///     lambdas: Regularisation parameter array.
///
/// Returns:
///     (residual_norms, solution_norms) tuple.
#[pyfunction]
#[pyo3(signature = (a, b, nrows, ncols, lambdas))]
fn tikhonov_lcurve(
    py: Python<'_>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray1<f64>,
    nrows: usize,
    ncols: usize,
    lambdas: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let a_arr = a.as_array();
    let a_flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| a_arr[[i, j]]))
        .collect();
    let b_s = b
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lam_s = lambdas
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (res, sol) = inverse_mod::tikhonov_lcurve(&a_flat, b_s, nrows, ncols, lam_s);
    Ok((
        res.into_pyarray(py).unbind(),
        sol.into_pyarray(py).unbind(),
    ))
}

/// Solve a Born-inversion problem with Tikhonov regularisation.
///
/// Args:
///     g_real: Real part of the Green's function matrix (nrows × ncols).
///     g_imag: Imaginary part (nrows × ncols).
///     y_real: Real part of measurement vector (length nrows).
///     y_imag: Imaginary part (length nrows).
///     nrows: Number of rows.
///     ncols: Number of columns.
///     lambda: Regularisation parameter.
///
/// Returns:
///     (real_solution, imag_solution) tuple.
#[pyfunction]
#[pyo3(signature = (g_real, g_imag, y_real, y_imag, nrows, ncols, lambda))]
fn born_inversion_regularized(
    py: Python<'_>,
    g_real: PyReadonlyArray2<f64>,
    g_imag: PyReadonlyArray2<f64>,
    y_real: PyReadonlyArray1<f64>,
    y_imag: PyReadonlyArray1<f64>,
    nrows: usize,
    ncols: usize,
    lambda: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let gr = g_real.as_array();
    let gi = g_imag.as_array();
    let gr_flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| gr[[i, j]]))
        .collect();
    let gi_flat: Vec<f64> = (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| gi[[i, j]]))
        .collect();
    let yr_s = y_real
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let yi_s = y_imag
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (re, im) = inverse_mod::born_inversion_regularized(
        &gr_flat, &gi_flat, yr_s, yi_s, nrows, ncols, lambda,
    );
    Ok((
        re.into_pyarray(py).unbind(),
        im.into_pyarray(py).unbind(),
    ))
}

/// Compute the adjoint-gradient convergence curve.
///
/// Args:
///     n_iter: Number of iterations.
///     initial_error: Initial relative error.
///     decay: Per-iteration error decay factor.
///
/// Returns:
///     Error array of length *n_iter*.
#[pyfunction]
#[pyo3(signature = (n_iter, initial_error, decay))]
fn adjoint_gradient_convergence(
    py: Python<'_>,
    n_iter: usize,
    initial_error: f64,
    decay: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = inverse_mod::adjoint_gradient_convergence(n_iter, initial_error, decay);
    Ok(result.into_pyarray(py).unbind())
}

// ============================================================================
// sonogenetics.rs
// ============================================================================

/// Compute the Hill activation probability for mechanosensitive channels.
///
/// P(p) = p^n / (p_threshold^n + p^n)
///
/// Args:
///     pressure_arr: Pressure amplitude array [Pa].
///     p_threshold_pa: Half-activation pressure [Pa].
///     hill_n: Hill coefficient.
///
/// Returns:
///     Activation probability array (0–1).
#[pyfunction]
#[pyo3(signature = (pressure_arr, p_threshold_pa, hill_n))]
fn hill_activation_probability(
    py: Python<'_>,
    pressure_arr: PyReadonlyArray1<f64>,
    p_threshold_pa: f64,
    hill_n: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = pressure_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = sonogenetics::hill_activation_probability(p_s, p_threshold_pa, hill_n);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the 1-D acoustic radiation force density.
///
/// F = 2 * alpha * I / c
///
/// Args:
///     intensity_w_m2: Intensity array [W/m²].
///     alpha_np_m: Attenuation [Np/m].
///     c: Sound speed [m/s].
///
/// Returns:
///     Radiation force density array [N/m³].
#[pyfunction]
#[pyo3(signature = (intensity_w_m2, alpha_np_m, c))]
fn radiation_force_1d(
    py: Python<'_>,
    intensity_w_m2: PyReadonlyArray1<f64>,
    alpha_np_m: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let i_s = intensity_w_m2
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = sonogenetics::radiation_force_1d(i_s, alpha_np_m, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the steady acoustic streaming velocity (Eckart streaming).
///
/// Args:
///     i_w_m2: Beam intensity [W/m²].
///     mu_pa_s: Dynamic viscosity [Pa·s].
///     alpha_np_m: Attenuation [Np/m].
///     c: Sound speed [m/s].
///     l_m: Beam propagation length [m].
///
/// Returns:
///     Streaming velocity [m/s].
#[pyfunction]
#[pyo3(signature = (i_w_m2, mu_pa_s, alpha_np_m, c, l_m))]
fn acoustic_streaming_velocity(
    i_w_m2: f64,
    mu_pa_s: f64,
    alpha_np_m: f64,
    c: f64,
    l_m: f64,
) -> PyResult<f64> {
    Ok(sonogenetics::acoustic_streaming_velocity(
        i_w_m2, mu_pa_s, alpha_np_m, c, l_m,
    ))
}

/// Compute in-situ spatial-peak time-average intensity (ISPTA).
///
/// Args:
///     p_pa: Pressure time series [Pa].
///     dt_s: Sample interval [s].
///     rho: Density [kg/m³].
///     c: Sound speed [m/s].
///
/// Returns:
///     ISPTA [W/cm²].
#[pyfunction]
#[pyo3(signature = (p_pa, dt_s, rho, c))]
fn ispta_w_cm2(
    p_pa: PyReadonlyArray1<f64>,
    dt_s: f64,
    rho: f64,
    c: f64,
) -> PyResult<f64> {
    let p_s = p_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(sonogenetics::ispta_w_cm2(p_s, dt_s, rho, c))
}

// ============================================================================
// rtm.rs
// ============================================================================

/// Compute a focused Gaussian beam field in 2-D including skull transmission.
///
/// Returns (real_field, imag_field) each of shape (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     x_f: Focus x-coordinate [m].
///     z_f: Focus z-coordinate [m].
///     freq_hz: Frequency [Hz].
///     c_brain: Brain sound speed [m/s].
///     w0_m: Beam waist [m].
///     skull_transmission_real: Real part of skull transmission coefficient.
///     skull_transmission_imag: Imaginary part.
///     r_back: Back-wall reflection coefficient.
///     z_back: Back-wall axial position [m].
///
/// Returns:
///     (real_nx_nz, imag_nx_nz) tuple of 2-D arrays.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, x_f, z_f, freq_hz, c_brain, w0_m, skull_transmission_real, skull_transmission_imag, r_back, z_back))]
fn focused_gaussian_beam_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    freq_hz: f64,
    c_brain: f64,
    w0_m: f64,
    skull_transmission_real: f64,
    skull_transmission_imag: f64,
    r_back: f64,
    z_back: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let skull_transmission = Complex64::new(skull_transmission_real, skull_transmission_imag);
    let (real_flat, imag_flat) = rtm_mod::focused_gaussian_beam_2d(
        x_s, z_s, x_f, z_f, freq_hz, c_brain, w0_m, skull_transmission, r_back, z_back,
    );
    let real_arr = Array2::from_shape_vec((nx, nz), real_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let imag_arr = Array2::from_shape_vec((nx, nz), imag_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        real_arr.into_pyarray(py).unbind(),
        imag_arr.into_pyarray(py).unbind(),
    ))
}

/// Compute the 2-D back-propagation Green's function.
///
/// Returns (real_field, imag_field) each of shape (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     x_f: Source x-coordinate [m].
///     z_f: Source z-coordinate [m].
///     k_br: Wave number in brain [rad/m].
///
/// Returns:
///     (real_nx_nz, imag_nx_nz) tuple.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, x_f, z_f, k_br))]
fn backprop_green_function_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    k_br: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let (real_flat, imag_flat) =
        rtm_mod::backprop_green_function_2d(x_s, z_s, x_f, z_f, k_br);
    let real_arr = Array2::from_shape_vec((nx, nz), real_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let imag_arr = Array2::from_shape_vec((nx, nz), imag_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        real_arr.into_pyarray(py).unbind(),
        imag_arr.into_pyarray(py).unbind(),
    ))
}

/// Apply the zero-lag cross-correlation imaging condition for RTM.
///
/// image[i,j] = sum_t p_fwd[i,j,t] * p_bwd[i,j,t]
///
/// This version operates on single-frequency snapshots (no time axis):
/// image[i,j] = Re(p_fwd[i,j] * conj(p_bwd[i,j]))
///
/// Args:
///     p_fwd_real: Real part of forward-propagated field (nx × nz).
///     p_fwd_imag: Imaginary part.
///     p_bwd_real: Real part of back-propagated field (nx × nz).
///     p_bwd_imag: Imaginary part.
///     nx: Number of lateral grid points.
///     nz: Number of axial grid points.
///
/// Returns:
///     Image ndarray of shape (nx, nz).
#[pyfunction]
#[pyo3(signature = (p_fwd_real, p_fwd_imag, p_bwd_real, p_bwd_imag, nx, nz))]
fn rtm_imaging_condition(
    py: Python<'_>,
    p_fwd_real: PyReadonlyArray2<f64>,
    p_fwd_imag: PyReadonlyArray2<f64>,
    p_bwd_real: PyReadonlyArray2<f64>,
    p_bwd_imag: PyReadonlyArray2<f64>,
    nx: usize,
    nz: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let fr = p_fwd_real.as_array();
    let fi = p_fwd_imag.as_array();
    let br = p_bwd_real.as_array();
    let bi = p_bwd_imag.as_array();
    let fr_flat: Vec<f64> = (0..nx).flat_map(|i| (0..nz).map(move |j| fr[[i, j]])).collect();
    let fi_flat: Vec<f64> = (0..nx).flat_map(|i| (0..nz).map(move |j| fi[[i, j]])).collect();
    let br_flat: Vec<f64> = (0..nx).flat_map(|i| (0..nz).map(move |j| br[[i, j]])).collect();
    let bi_flat: Vec<f64> = (0..nx).flat_map(|i| (0..nz).map(move |j| bi[[i, j]])).collect();
    let flat = rtm_mod::rtm_imaging_condition(&fr_flat, &fi_flat, &br_flat, &bi_flat, nx, nz);
    let arr2d = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Fuse multiple single-frequency RTM images by coherent averaging.
///
/// Args:
///     images: List of PyReadonlyArray2 images, each of shape (nx, nz).
///
/// Returns:
///     Fused image of shape (nx, nz).
#[pyfunction]
#[pyo3(signature = (images,))]
fn rtm_multi_frequency_fusion(
    py: Python<'_>,
    images: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    if images.is_empty() {
        return Err(PyRuntimeError::new_err("images list must not be empty"));
    }
    let first = images[0].as_array();
    let (nx, nz) = first.dim();
    let vecs: Vec<Vec<f64>> = images
        .iter()
        .map(|img| {
            let arr = img.as_array();
            (0..nx).flat_map(|i| (0..nz).map(move |j| arr[[i, j]])).collect()
        })
        .collect();
    let flat = rtm_mod::rtm_multi_frequency_fusion(&vecs);
    let arr2d = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Compute temporal modulation frequencies for transcranial standing-wave suppression.
///
/// Args:
///     f0_hz: Carrier frequency [Hz].
///     m_steps: Number of modulation steps.
///     c: Sound speed [m/s].
///     d_back_m: Back-wall distance [m].
///
/// Returns:
///     Modulation frequency array [Hz].
#[pyfunction]
#[pyo3(signature = (f0_hz, m_steps, c, d_back_m))]
fn temporal_modulation_frequencies(
    py: Python<'_>,
    f0_hz: f64,
    m_steps: usize,
    c: f64,
    d_back_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = rtm_mod::temporal_modulation_frequencies(f0_hz, m_steps, c, d_back_m);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the standing-wave suppression gain factor.
///
/// G = 1 - |r_back|²
///
/// Args:
///     r_back: Back-wall reflection coefficient magnitude.
///
/// Returns:
///     Suppression gain factor (0–1).
#[pyfunction]
#[pyo3(signature = (r_back,))]
fn standing_wave_suppression_gain(r_back: f64) -> PyResult<f64> {
    Ok(rtm_mod::standing_wave_suppression_gain(r_back))
}

// ============================================================================
// Registration
// ============================================================================

/// Register all book-physics functions into the given Python sub-module.
pub fn register_book(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // wave
    m.add_function(wrap_pyfunction!(standing_wave_1d, m)?)?;
    m.add_function(wrap_pyfunction!(plane_wave_pressure_1d, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_wave_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(reflection_pressure_coeff, m)?)?;
    m.add_function(wrap_pyfunction!(transmission_pressure_coeff, m)?)?;
    m.add_function(wrap_pyfunction!(power_law_attenuation_np_m, m)?)?;
    m.add_function(wrap_pyfunction!(absorption_power_law_db_cm, m)?)?;
    m.add_function(wrap_pyfunction!(fdtd_phase_error_1d, m)?)?;
    m.add_function(wrap_pyfunction!(pstd_phase_error, m)?)?;
    m.add_function(wrap_pyfunction!(kspace_correction_error, m)?)?;
    m.add_function(wrap_pyfunction!(fdtd_cfl_limit, m)?)?;
    m.add_function(wrap_pyfunction!(fubini_harmonic_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(fubini_harmonic_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(shock_formation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(westervelt_harmonic_evolution, m)?)?;
    // transducer
    m.add_function(wrap_pyfunction!(circular_piston_directivity, m)?)?;
    m.add_function(wrap_pyfunction!(linear_array_factor, m)?)?;
    m.add_function(wrap_pyfunction!(grating_lobe_angles, m)?)?;
    m.add_function(wrap_pyfunction!(apodization_weights, m)?)?;
    m.add_function(wrap_pyfunction!(delay_law_focus_2d, m)?)?;
    m.add_function(wrap_pyfunction!(beam_pattern_2d, m)?)?;
    m.add_function(wrap_pyfunction!(circular_piston_onaxis, m)?)?;
    m.add_function(wrap_pyfunction!(focused_bowl_onaxis, m)?)?;
    m.add_function(wrap_pyfunction!(bli_stencil_weights, m)?)?;
    // cavitation
    m.add_function(wrap_pyfunction!(minnaert_resonance_hz, m)?)?;
    m.add_function(wrap_pyfunction!(blake_threshold_pa, m)?)?;
    m.add_function(wrap_pyfunction!(rayleigh_collapse_time_s, m)?)?;
    m.add_function(wrap_pyfunction!(rayleigh_plesset_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(keller_miksis_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(bubble_power_spectrum, m)?)?;
    // tissue
    m.add_function(wrap_pyfunction!(water_sound_speed_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(water_density_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(ba_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(tissue_absorption_db_cm, m)?)?;
    m.add_function(wrap_pyfunction!(kramers_kronig_sound_speed, m)?)?;
    m.add_function(wrap_pyfunction!(tissue_properties, m)?)?;
    // safety
    m.add_function(wrap_pyfunction!(mechanical_index, m)?)?;
    m.add_function(wrap_pyfunction!(thermal_index_soft_tissue, m)?)?;
    m.add_function(wrap_pyfunction!(thermal_index_bone, m)?)?;
    m.add_function(wrap_pyfunction!(cem43_cumulative, m)?)?;
    m.add_function(wrap_pyfunction!(arrhenius_damage_integral, m)?)?;
    m.add_function(wrap_pyfunction!(fda_ispta_limit_mw_cm2, m)?)?;
    m.add_function(wrap_pyfunction!(fda_isppa_limit_w_cm2, m)?)?;
    // skull
    m.add_function(wrap_pyfunction!(skull_insertion_loss_two_way_db, m)?)?;
    m.add_function(wrap_pyfunction!(skull_phase_screen, m)?)?;
    m.add_function(wrap_pyfunction!(hu_to_sound_speed_schneider, m)?)?;
    m.add_function(wrap_pyfunction!(hu_to_density_schneider, m)?)?;
    m.add_function(wrap_pyfunction!(strehl_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(skull_surface_temperature_rise, m)?)?;
    m.add_function(wrap_pyfunction!(skull_transfer_matrix_transmission, m)?)?;
    m.add_function(wrap_pyfunction!(skull_transmission_spectrum, m)?)?;
    // photoacoustics
    m.add_function(wrap_pyfunction!(hbo2_molar_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(hb_molar_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(gruneisen_parameter_water, m)?)?;
    m.add_function(wrap_pyfunction!(pa_sphere_pressure_signal, m)?)?;
    m.add_function(wrap_pyfunction!(pa_axial_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(spectroscopic_unmixing_lstsq, m)?)?;
    // elastography
    m.add_function(wrap_pyfunction!(shear_wave_speed, m)?)?;
    m.add_function(wrap_pyfunction!(voigt_complex_modulus, m)?)?;
    m.add_function(wrap_pyfunction!(springpot_complex_modulus, m)?)?;
    m.add_function(wrap_pyfunction!(voigt_shear_wave_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(mre_displacement_field, m)?)?;
    // imaging
    m.add_function(wrap_pyfunction!(lateral_psf_sinc2, m)?)?;
    m.add_function(wrap_pyfunction!(axial_psf_rect, m)?)?;
    m.add_function(wrap_pyfunction!(doppler_frequency_shift, m)?)?;
    m.add_function(wrap_pyfunction!(pw_compounding_lateral_psf, m)?)?;
    m.add_function(wrap_pyfunction!(lateral_resolution_m, m)?)?;
    // thermal
    m.add_function(wrap_pyfunction!(bioheat_focal_temperature_rise, m)?)?;
    m.add_function(wrap_pyfunction!(hifu_focal_pressure_gain, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_power_deposition_2d, m)?)?;
    // inverse
    m.add_function(wrap_pyfunction!(helmholtz_1d_fd_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_singular_values, m)?)?;
    m.add_function(wrap_pyfunction!(tikhonov_lcurve, m)?)?;
    m.add_function(wrap_pyfunction!(born_inversion_regularized, m)?)?;
    m.add_function(wrap_pyfunction!(adjoint_gradient_convergence, m)?)?;
    // sonogenetics
    m.add_function(wrap_pyfunction!(hill_activation_probability, m)?)?;
    m.add_function(wrap_pyfunction!(radiation_force_1d, m)?)?;
    m.add_function(wrap_pyfunction!(acoustic_streaming_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(ispta_w_cm2, m)?)?;
    // rtm
    m.add_function(wrap_pyfunction!(focused_gaussian_beam_2d, m)?)?;
    m.add_function(wrap_pyfunction!(backprop_green_function_2d, m)?)?;
    m.add_function(wrap_pyfunction!(rtm_imaging_condition, m)?)?;
    m.add_function(wrap_pyfunction!(rtm_multi_frequency_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_modulation_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(standing_wave_suppression_gain, m)?)?;
    Ok(())
}
