//! PyO3 wrapper for the Keller–Miksis integrator (compressible-liquid bubble
//! dynamics).
//!
//! This module is a thin binding only: the Keller–Miksis ODE physics lives in
//! `kwavers_physics::analytical::cavitation::keller_miksis_shelled_rk4` (the single
//! source of truth). Here we only marshal the Python call: build the uniform time
//! grid, delegate, and return numpy arrays.
//!
//! The shell viscosity `xi_s` [Pa·s·m] selects the encapsulated-microbubble shell
//! damping (de Jong 1994 lumped linear model): `xi_s = 0` is the bare gas bubble,
//! `xi_s > 0` adds the `−4 ξ_s Ṙ / R²` shell-damping wall pressure. Both cases are
//! the same canonical physics function (the bare case is `xi_s = 0`), so no model
//! branching or approximation happens in the binding.
//!
//! # References
//! - Keller & Miksis (1980) J. Acoust. Soc. Am. 68(2):628.
//! - de Jong et al. (1994) Ultrasonics 32:447 (linear shell viscosity).

use ndarray::Array1;
use numpy::{ToPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Integrate the Keller-Miksis equation using fixed-step RK4 (delegates to
/// `kwavers_physics`). Extends Rayleigh-Plesset with the liquid-compressibility
/// (Mach-number) correction.
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
/// c_l : liquid sound speed [m/s]
/// xi_s : encapsulating-shell viscosity [Pa·s·m] (de Jong 1994 lumped linear shell).
///        0 = bare gas bubble; > 0 = coated/encapsulated microbubble.
///
/// Returns
/// -------
/// (time_s, radius_m, rdot_m_s) as numpy arrays of length n_steps+1.
#[pyfunction]
#[pyo3(signature = (
    r0_m, rdot0_m_s, p_inf_pa, p_ac_pa, frequency_hz,
    t_end_s, n_steps, rho, sigma, gamma, mu, pv_pa, c_l,
    xi_s = 0.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_keller_miksis(
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
    c_l: f64,
    xi_s: f64,
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
    if rho <= 0.0 || c_l <= 0.0 {
        return Err(PyValueError::new_err("rho and c_l must be > 0"));
    }
    if xi_s < 0.0 {
        return Err(PyValueError::new_err("xi_s must be >= 0"));
    }

    let dt = t_end_s / n_steps as f64;
    let time: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();
    // Single source of truth: the bare bubble (xi_s = 0) and the coated
    // microbubble (xi_s > 0) are the SAME canonical Keller–Miksis integrator;
    // the shell term −4 ξ_s Ṙ / R² vanishes identically at xi_s = 0.
    let (radius, rdot) = kwavers_physics::analytical::cavitation::keller_miksis_shelled_rk4(
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
        xi_s,
        c_l,
    );
    Ok((
        Array1::from(time).to_pyarray(py).into(),
        Array1::from(radius).to_pyarray(py).into(),
        Array1::from(rdot).to_pyarray(py).into(),
    ))
}

/// Integrate the Keller–Herring equation using the same Rust-backed conservative
/// kernel as `solve_keller_miksis`.
///
/// Parameters, validation, and return shape are intentionally identical to
/// `solve_keller_miksis`; KH is currently represented as a typed contract variant
/// around the same adaptive kernel until a dedicated KH correction is added.
#[pyfunction]
#[pyo3(signature = (
    r0_m, rdot0_m_s, p_inf_pa, p_ac_pa, frequency_hz,
    t_end_s, n_steps, rho, sigma, gamma, mu, pv_pa, c_l,
    xi_s = 0.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_keller_herring(
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
    c_l: f64,
    xi_s: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    solve_keller_miksis(
        py,
        r0_m,
        rdot0_m_s,
        p_inf_pa,
        p_ac_pa,
        frequency_hz,
        t_end_s,
        n_steps,
        rho,
        sigma,
        gamma,
        mu,
        pv_pa,
        c_l,
        xi_s,
    )
}

