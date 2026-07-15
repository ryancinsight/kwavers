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
//! Integration is by the rectangle rule (kwavers_physics). Cell death probability follows
//! `P_death = 1 − exp(−Ω)`. Typical human tissue values:
//! `Ea ≈ 2.77 × 10⁵ J/mol`, `A ≈ 3.1 × 10⁴³ s⁻¹` (Henriques 1947).
//!
//! # References
//!
//! - Henriques & Moritz (1947) Am. J. Pathol. 23:695

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const KELVIN_OFFSET_C: f64 = 273.15; // K↔°C boundary conversion

/// Compute the cumulative Arrhenius thermal damage integral Ω(t).
///
/// Thin wrapper: the Arrhenius damage physics lives in
/// `kwavers_physics::analytical::safety::arrhenius_cumulative` (single source of
/// truth). Here we only convert K→°C and derive the time step before delegating.
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
    if n < 2 {
        return Ok(Array1::<f64>::zeros(n).to_pyarray(py).into());
    }
    let dt_s = (times[n - 1] - times[0]) / (n - 1) as f64; // uniform step
    let celsius: Vec<f64> = temps.iter().map(|&t_k| t_k - KELVIN_OFFSET_C).collect();
    let omega = kwavers_physics::analytical::safety::arrhenius_cumulative(
        &celsius,
        dt_s,
        a_hz,
        ea_j_per_mol,
    );
    Ok(Array1::from(omega).to_pyarray(py).into())
}
