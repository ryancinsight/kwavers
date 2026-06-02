//! Rayleigh-Plesset ODE integration (incompressible bubble dynamics).
//!
//! # Theorem
//!
//! The Rayleigh-Plesset equation (Rayleigh 1917, Plesset 1949) describes the
//! radial dynamics of a spherical gas bubble in an incompressible liquid:
//!
//! ```text
//! R R̈ + (3/2) Ṙ² = [p_G(R) − p_∞ − p_ac(t) − 4μṘ/R − 2σ/R] / ρ
//! ```
//!
//! where the polytropic gas pressure is:
//!
//! ```text
//! p_G(R) = (p_∞ + 2σ/R₀ − p_v)(R₀/R)^{3γ} + p_v
//! ```
//!
//! Rewritten as a first-order system `y = [R, Ṙ]` and integrated by RK4.
//!
//! # References
//!
//! - Rayleigh (1917) Phil. Mag. 34:94
//! - Plesset (1949) J. Appl. Mech. 16:277

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[inline]
pub(super) fn rp_rhs(
    t: f64,
    r: f64,
    rdot: f64,
    r0: f64,
    p_inf: f64,
    p_ac: f64,
    omega: f64,
    rho: f64,
    sigma: f64,
    gamma: f64,
    mu: f64,
    pv: f64,
) -> (f64, f64) {
    let r_safe = r.max(1.0e-12);
    let p_g0 = p_inf + 2.0 * sigma / r0 - pv;
    let p_g = p_g0 * (r0 / r_safe).powf(3.0 * gamma) + pv;
    let p_ac_t = p_ac * (omega * t).sin();
    let numerator = (p_g - p_inf - p_ac_t - 4.0 * mu * rdot / r_safe - 2.0 * sigma / r_safe) / rho;
    let rddot = numerator / r_safe - 1.5 * rdot * rdot / r_safe;
    (rdot, rddot)
}

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
pub fn solve_rayleigh_plesset<'py>(
    py: Python<'py>,
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

    let n_out = n_steps + 1;
    let mut time = Array1::<f64>::zeros(n_out);
    let mut radius = Array1::<f64>::zeros(n_out);
    let mut rdot = Array1::<f64>::zeros(n_out);
    let dt = t_end_s / n_steps as f64;
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;

    time[0] = 0.0;
    radius[0] = r0_m;
    rdot[0] = rdot0_m_s;
    let mut r_cur = r0_m;
    let mut v_cur = rdot0_m_s;

    for i in 1..n_out {
        let t_cur = (i - 1) as f64 * dt;
        let (dr1, dv1) = rp_rhs(
            t_cur, r_cur, v_cur, r0_m, p_inf_pa, p_ac_pa, omega, rho, sigma, gamma, mu, pv_pa,
        );
        let (dr2, dv2) = rp_rhs(
            t_cur + 0.5 * dt,
            r_cur + 0.5 * dt * dr1,
            v_cur + 0.5 * dt * dv1,
            r0_m,
            p_inf_pa,
            p_ac_pa,
            omega,
            rho,
            sigma,
            gamma,
            mu,
            pv_pa,
        );
        let (dr3, dv3) = rp_rhs(
            t_cur + 0.5 * dt,
            r_cur + 0.5 * dt * dr2,
            v_cur + 0.5 * dt * dv2,
            r0_m,
            p_inf_pa,
            p_ac_pa,
            omega,
            rho,
            sigma,
            gamma,
            mu,
            pv_pa,
        );
        let (dr4, dv4) = rp_rhs(
            t_cur + dt,
            r_cur + dt * dr3,
            v_cur + dt * dv3,
            r0_m,
            p_inf_pa,
            p_ac_pa,
            omega,
            rho,
            sigma,
            gamma,
            mu,
            pv_pa,
        );

        r_cur += dt / 6.0 * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4);
        v_cur += dt / 6.0 * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
        r_cur = r_cur.max(1.0e-12);

        time[i] = t_cur + dt;
        radius[i] = r_cur;
        rdot[i] = v_cur;
    }

    Ok((
        time.into_pyarray(py).into(),
        radius.into_pyarray(py).into(),
        rdot.into_pyarray(py).into(),
    ))
}
