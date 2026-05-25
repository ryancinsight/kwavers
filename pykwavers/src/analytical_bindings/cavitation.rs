//! PyO3 bindings for `kwavers::physics::analytical::cavitation`.

use kwavers::physics::analytical::cavitation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
pub fn minnaert_resonance_hz(r0_m: f64, gamma: f64, p0_pa: f64, rho: f64) -> PyResult<f64> {
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
pub fn blake_threshold_pa(r0_m: f64, p0_pa: f64, sigma_n_m: f64) -> PyResult<f64> {
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
pub fn rayleigh_collapse_time_s(rmax_m: f64, p_inf_pa: f64, rho: f64) -> PyResult<f64> {
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
pub fn rayleigh_plesset_rk4(
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
    Ok((r.into_pyarray(py).unbind(), rdot.into_pyarray(py).unbind()))
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
pub fn keller_miksis_rk4(
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
    Ok((r.into_pyarray(py).unbind(), rdot.into_pyarray(py).unbind()))
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
