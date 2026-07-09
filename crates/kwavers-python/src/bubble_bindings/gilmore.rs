//! PyO3 wrapper for the Gilmore–Tait integrator (high-amplitude, compressible
//! bubble dynamics).
//!
//! Thin binding only: the Gilmore ODE physics lives in
//! `kwavers_physics::acoustics::bubble_dynamics::gilmore::GilmoreSolver`
//! (the single source of truth). Here we build the bubble parameters, run the
//! solver's fixed-step RK4 loop, and return numpy arrays. The solver applies the
//! sinusoidal drive `p_ac·sin(ω t)` internally (ω = `frequency_hz`), so the
//! amplitude `p_ac_pa` is passed unmodified each step.
//!
//! # Reference
//! Gilmore, F. R. (1952), Caltech Hydrodynamics Lab Report 26-4.

use kwavers_physics::acoustics::bubble_dynamics::gilmore::GilmoreSolver;
use kwavers_physics::acoustics::bubble_dynamics::{BubbleParameters, BubbleState};
use leto::Array1;
use numpy::{ToPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Integrate the Gilmore–Tait equation with fixed-step RK4 (delegates to
/// `GilmoreSolver`). More accurate than Keller–Miksis for violent collapse
/// (wall Mach > 0.1); the liquid compressibility uses the Tait equation of
/// state. The polytropic exponent is taken from the (air) gas species, so
/// `gamma` is accepted only for signature parity with `solve_keller_miksis`.
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
/// gamma : polytropic index (accepted for parity; solver uses the gas species)
/// mu : liquid dynamic viscosity [Pa·s]
/// pv_pa : vapour pressure [Pa]
/// c_l : liquid sound speed [m/s]
///
/// Returns
/// -------
/// (time_s, radius_m, rdot_m_s) as numpy arrays of length n_steps+1.
#[pyfunction]
#[pyo3(signature = (
    r0_m, rdot0_m_s, p_inf_pa, p_ac_pa, frequency_hz,
    t_end_s, n_steps, rho, sigma, gamma, mu, pv_pa, c_l,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_gilmore(
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

    let params = BubbleParameters {
        r0: r0_m,
        p0: p_inf_pa,
        rho_liquid: rho,
        c_liquid: c_l,
        mu_liquid: mu,
        sigma,
        pv: pv_pa,
        gamma,
        driving_frequency: frequency_hz,
        driving_amplitude: p_ac_pa,
        ..BubbleParameters::default()
    };

    // Initial state borrows params; the solver then takes ownership.
    let mut state = BubbleState::new(&params);
    state.wall_velocity = rdot0_m_s;
    let solver = GilmoreSolver::new(params);

    let dt = t_end_s / n_steps as f64;
    let mut time = Vec::with_capacity(n_steps + 1);
    let mut radius = Vec::with_capacity(n_steps + 1);
    let mut rdot = Vec::with_capacity(n_steps + 1);
    time.push(0.0);
    radius.push(state.radius);
    rdot.push(state.wall_velocity);

    for i in 0..n_steps {
        let t = i as f64 * dt;
        state = solver.step_rk4(&state, p_ac_pa, t, dt);
        time.push((i + 1) as f64 * dt);
        radius.push(state.radius);
        rdot.push(state.wall_velocity);
    }

    Ok((
        Array1::from(time).to_pyarray(py).into(),
        Array1::from(radius).to_pyarray(py).into(),
        Array1::from(rdot).to_pyarray(py).into(),
    ))
}

