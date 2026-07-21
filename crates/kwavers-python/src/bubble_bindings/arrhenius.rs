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
//! Integration is by the rectangle rule (Asclepius). Cell death probability follows
//! `P_death = 1 − exp(−Ω)`. Typical human tissue values:
//! `Ea ≈ 2.77 × 10⁵ J/mol`, `A ≈ 3.1 × 10⁴³ s⁻¹` (Henriques 1947).
//!
//! # References
//!
//! - Henriques & Moritz (1947) Am. J. Pathol. 23:695

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Compute the cumulative Arrhenius thermal damage integral Ω(t).
///
/// Asclepius owns the Arrhenius law. This binding validates array shape and
/// derives the uniform time step.
///
/// Parameters
/// ----------
/// temperatures_k : temperature array [K] at each time point.
/// times_s : uniformly-spaced time array [s], same length as temperatures_k.
/// ea_j_per_mol : activation energy Ea [J/mol].
/// a_hz : frequency factor A [s⁻¹].
///
/// Returns
/// -------
/// 1-D array of cumulative Ω at each time point.
#[pyfunction]
pub fn compute_arrhenius_damage(
    py: Python<'_>,
    temperatures_k: PyReadonlyArray1<f64>,
    times_s: PyReadonlyArray1<f64>,
    ea_j_per_mol: f64,
    a_hz: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let temps = temperatures_k
        .as_slice()
        .map_err(|source| PyRuntimeError::new_err(source.to_string()))?;
    let times = times_s
        .as_slice()
        .map_err(|source| PyRuntimeError::new_err(source.to_string()))?;
    if temps.len() != times.len() {
        return Err(PyValueError::new_err(
            "temperatures_k and times_s must have the same length",
        ));
    }
    let n = temps.len();
    if n < 2 {
        return Err(PyValueError::new_err(
            "temperatures_k and times_s must contain at least two samples",
        ));
    }
    let dt_s = (times[n - 1] - times[0]) / (n - 1) as f64; // uniform step
    let temperatures = temps.to_vec();
    let omega = py
        .detach(move || {
            crate::response::thermal::arrhenius_cumulative_kelvin(
                &temperatures,
                dt_s,
                a_hz,
                ea_j_per_mol,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(omega.to_pyarray(py).into())
}
