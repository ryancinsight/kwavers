//! Keller-Miksis ODE integration (compressible bubble dynamics).
//!
//! # Theorem
//!
//! The Keller-Miksis equation (Keller & Miksis 1980) extends Rayleigh-Plesset
//! with first-order liquid compressibility corrections:
//!
//! ```text
//! (1 − Ṙ/c) R R̈ + (3/2)(1 − Ṙ/(3c)) Ṙ² =
//!     (1 + Ṙ/c)/ρ · [p_L − p_∞ − p_ac·sin(ωt)]
//!     + R/(ρ·c) · ṗ_L
//! ```
//!
//! where:
//!
//! ```text
//! p_L(R) = (p_∞ + 2σ/R₀)(R₀/R)^{3κ} − 2σ/R − 4μṘ/R − 4ξṘ/R²
//! ṗ_L ≈ −3κ p_gas Ṙ/R   (dominant gas compression term)
//! ```
//!
//! The shell viscosity term `4ξṘ/R²` models encapsulated microbubble shells;
//! set `xi_s = 0` for a bare gas bubble.
//!
//! The Mach number `Ṙ/c` is clamped to `[−0.99, 0.99]` to prevent
//! division by zero near catastrophic collapse (|Ṙ| → c).
//!
//! # References
//!
//! - Keller & Miksis (1980) J. Acoust. Soc. Am. 68(2):628

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn km_rhs(
    t: f64,
    r: f64,
    rdot: f64,
    r0: f64,
    p_inf: f64,
    p_ac: f64,
    omega: f64,
    rho: f64,
    sigma: f64,
    kappa: f64,
    mu: f64,
    xi_s: f64,
    c_l: f64,
) -> (f64, f64) {
    let r_safe = r.max(1.0e-12);
    let p_gas = (p_inf + 2.0 * sigma / r0) * (r0 / r_safe).powf(3.0 * kappa);
    let p_l = p_gas
        - 2.0 * sigma / r_safe
        - 4.0 * mu * rdot / r_safe
        - 4.0 * xi_s * rdot / (r_safe * r_safe);
    let dp_l_dt = -3.0 * kappa * p_gas * rdot / r_safe;
    let p_drive = p_inf + p_ac * (omega * t).sin();

    let mach = (rdot / c_l).clamp(-0.99, 0.99);
    let lhs_r_coeff = (1.0 - mach) * r_safe;
    if lhs_r_coeff.abs() < 1.0e-20 {
        return (rdot, 0.0);
    }

    let rhs = (1.0 + mach) * (p_l - p_drive) / rho + r_safe * dp_l_dt / (rho * c_l)
        - 1.5 * (1.0 - mach / 3.0) * rdot * rdot;

    let rddot = rhs / lhs_r_coeff;
    (rdot, rddot)
}

/// Integrate the Keller-Miksis equation using fixed-step RK4.
///
/// Extends Rayleigh-Plesset with liquid compressibility (Mach-number) correction.
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
/// pv_pa : vapour pressure [Pa] (accepted for API symmetry; unused in K-M)
/// c_l : liquid sound speed [m/s]
/// xi_s : shell viscosity parameter [Pa·s·m] (0 for bare bubble)
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
pub fn solve_keller_miksis<'py>(
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
    c_l: f64,
    xi_s: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let _ = pv_pa;
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

    let n_out = n_steps + 1;
    let mut time = Array1::<f64>::zeros(n_out);
    let mut radius = Array1::<f64>::zeros(n_out);
    let mut rdot_out = Array1::<f64>::zeros(n_out);

    let dt = t_end_s / n_steps as f64;
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;

    time[0] = 0.0;
    radius[0] = r0_m;
    rdot_out[0] = rdot0_m_s;

    let mut r_cur = r0_m;
    let mut v_cur = rdot0_m_s;

    for i in 1..n_out {
        let t_cur = (i - 1) as f64 * dt;

        let (dr1, dv1) = km_rhs(
            t_cur, r_cur, v_cur, r0_m, p_inf_pa, p_ac_pa, omega, rho, sigma, gamma, mu, xi_s, c_l,
        );
        let (dr2, dv2) = km_rhs(
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
            xi_s,
            c_l,
        );
        let (dr3, dv3) = km_rhs(
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
            xi_s,
            c_l,
        );
        let (dr4, dv4) = km_rhs(
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
            xi_s,
            c_l,
        );

        r_cur += dt / 6.0 * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4);
        v_cur += dt / 6.0 * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
        r_cur = r_cur.max(1.0e-12);

        time[i] = t_cur + dt;
        radius[i] = r_cur;
        rdot_out[i] = v_cur;
    }

    Ok((
        time.into_pyarray(py).into(),
        radius.into_pyarray(py).into(),
        rdot_out.into_pyarray(py).into(),
    ))
}
