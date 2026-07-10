//! PyO3 wrapper for the Rayleigh–Plesset integrator.
//!
//! This module is a thin binding only: the Rayleigh–Plesset ODE physics (RK4
//! integration, polytropic gas closure, surface-tension / viscous wall stress) lives
//! in `kwavers_physics::analytical::cavitation::rayleigh_plesset_rk4` (the single
//! source of truth). Here we only marshal the Python call: build the uniform time
//! grid, delegate, and return numpy arrays. This is the sole Rayleigh–Plesset
//! binding; the former raw-`t_arr` analytical binding was consolidated into it.
//!
//! # References
//! - Rayleigh (1917) Phil. Mag. 34:94; Plesset (1949) J. Appl. Mech. 16:277.

use numpy::ndarray::Array1;
use numpy::{ToPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Integrate the Rayleigh-Plesset equation using fixed-step RK4.
///
/// Parameters
/// ----------
/// r0_m : equilibrium bubble radius [m]
/// rdot0_m_s : initial wall velocity [m/s]
/// p_inf_pa : ambient far-field pressure [Pa]
/// p_ac_pa : acoustic driving pressure amplitude [Pa]
/// frequency_hz : driving frequency [Hz]
/// t_end_s : integration end time [s]
/// n_steps : number of RK4 steps
/// rho : liquid density [kg/m³]
/// sigma : surface tension [N/m]
/// gamma : polytropic index (dimensionless)
/// mu : liquid dynamic viscosity [Pa·s]
/// pv_pa : vapour pressure [Pa]
///
/// Returns
/// -------
/// (time_s, radius_m, rdot_m_s) as numpy arrays of length n_steps+1.
#[pyfunction]
#[pyo3(signature = (
    r0_m, rdot0_m_s, p_inf_pa, p_ac_pa, frequency_hz,
    t_end_s, n_steps, rho, sigma, gamma, mu, pv_pa
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_rayleigh_plesset(
    py: Python<'_>,
    r0_m: f64,
    rdot0_m_s: f64,
    p_inf_pa: f64,
    p_ac_pa: f64,
    frequency_hz: f64,
    t_end_s: f64,
    n_steps: usize,
    rho: f64,
    sigma: f64,
    gamma: f64,
    mu: f64,
    pv_pa: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    if r0_m <= 0.0 {
        return Err(PyValueError::new_err("r0_m must be > 0"));
    }
    if t_end_s <= 0.0 {
        return Err(PyValueError::new_err("t_end_s must be > 0"));
    }
    if n_steps == 0 {
        return Err(PyValueError::new_err("n_steps must be > 0"));
    }
    if rho <= 0.0 {
        return Err(PyValueError::new_err("rho must be > 0"));
    }

    let dt = t_end_s / n_steps as f64;
    let time: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();
    let (radius, rdot) = kwavers_physics::analytical::cavitation::rayleigh_plesset_rk4(
        r0_m,
        rdot0_m_s,
        p_ac_pa,
        frequency_hz,
        &time,
        p_inf_pa,
        rho,
        sigma,
        mu,
        gamma,
        pv_pa,
    );
    Ok((
        Array1::from(time).to_pyarray(py).into(),
        Array1::from(radius).to_pyarray(py).into(),
        Array1::from(rdot).to_pyarray(py).into(),
    ))
}

