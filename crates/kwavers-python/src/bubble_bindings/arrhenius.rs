//! Arrhenius thermal damage integral (Henriques & Moritz 1947).
//!
//! # Theorem
//!
//! The Arrhenius damage integral Ω(t) quantifies irreversible protein denaturation:
//!
//! ```text
//! dΩ/dt = A · exp(−Ea / (R_gas · T(t)))
//! ```
//!
//! where:
//! - `A` [s⁻¹] is the frequency factor (pre-exponential)
//! - `Ea` [J/mol] is the activation energy
//! - `R_gas = 8.314 J/(mol·K)` is the universal gas constant
//! - `T(t)` [K] is the instantaneous temperature
//!
//! Integration is by the trapezoidal rule. Cell death probability follows
//! `P_death = 1 − exp(−Ω)`. Typical human tissue values:
//! `Ea ≈ 2.77 × 10⁵ J/mol`, `A ≈ 3.1 × 10⁴³ s⁻¹` (Henriques 1947).
//!
//! # References
//!
//! - Henriques & Moritz (1947) Am. J. Pathol. 23:695

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const R_GAS: f64 = 8.314; // J/(mol·K)

/// Compute the cumulative Arrhenius thermal damage integral Ω(t).
///
/// `dΩ/dt = A · exp(−Ea / (R_gas · T(t)))`
///
/// Integration uses the trapezoidal rule over the provided time series.
///
/// Parameters
/// ----------
/// temperatures_k : temperature array [K] at each time point.
/// times_s : time array [s], same length as temperatures_k. Must be monotone.
/// ea_j_per_mol : activation energy Ea [J/mol].
/// a_hz : frequency factor A [s⁻¹].
///
/// Returns
/// -------
/// 1-D array of cumulative Ω at each time point.
#[pyfunction]
pub fn compute_arrhenius_damage<'py>(
    py: Python<'py>,
    temperatures_k: PyReadonlyArray1<f64>,
    times_s: PyReadonlyArray1<f64>,
    ea_j_per_mol: f64,
    a_hz: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let temps = temperatures_k.as_array();
    let times = times_s.as_array();
    if temps.len() != times.len() {
        return Err(PyValueError::new_err(
            "temperatures_k and times_s must have the same length",
        ));
    }
    if ea_j_per_mol < 0.0 {
        return Err(PyValueError::new_err("ea_j_per_mol must be >= 0"));
    }
    if a_hz < 0.0 {
        return Err(PyValueError::new_err("a_hz must be >= 0"));
    }
    let n = temps.len();
    let mut omega = Array1::<f64>::zeros(n);
    let rate = |t_k: f64| -> f64 {
        if t_k <= 0.0 {
            0.0
        } else {
            a_hz * (-ea_j_per_mol / (R_GAS * t_k)).exp()
        }
    };
    for i in 1..n {
        let dt = times[i] - times[i - 1];
        let increment = 0.5 * (rate(temps[i - 1]) + rate(temps[i])) * dt;
        omega[i] = omega[i - 1] + increment.max(0.0);
    }
    Ok(omega.into_pyarray(py).into())
}
