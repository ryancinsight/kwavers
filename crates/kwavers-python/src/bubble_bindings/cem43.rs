//! CEM43 thermal dose (Sapareto & Dewey 1984).
//!
//! # Theorem
//!
//! The cumulative equivalent minutes at 43 °C (CEM43) is:
//!
//! ```text
//! CEM43 = Σ_i R(T_i)^{43 − T_i} · Δt / 60
//! ```
//!
//! where `R = 0.5` if `T ≥ 43 °C` and `R = 0.25` if `T < 43 °C`, and
//! `Δt` is the time step in seconds. Division by 60 converts seconds to minutes.
//!
//! The threshold R = 0.5 reflects that a 1 °C increase above 43 °C halves the
//! required exposure time for equivalent thermal damage; below 43 °C the
//! relationship is weaker (R = 0.25). This semi-empirical model was validated
//! against cell-survival data for temperatures 41–57 °C.
//!
//! # References
//!
//! - Sapareto & Dewey (1984) Int. J. Radiat. Oncol. Biol. Phys. 10(6):787

use kwavers_physics::analytical::safety::cem43_cumulative;
use numpy::ndarray::Array1;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute the cumulative CEM43 thermal dose. Thin wrapper over
/// `kwavers_physics::analytical::safety::cem43_cumulative` (single source of truth);
/// the total dose is the last cumulative value.
///
/// Parameters
/// ----------
/// temperatures_c : array of temperatures in °C at each time step.
/// dt_s : constant time step between samples [s].
///
/// Returns
/// -------
/// Total CEM43 dose [min].
#[pyfunction]
pub fn compute_cem43(temperatures_c: PyReadonlyArray1<f64>, dt_s: f64) -> PyResult<f64> {
    if dt_s <= 0.0 {
        return Err(PyValueError::new_err("dt_s must be > 0"));
    }
    let temps = temperatures_c.as_array();
    let cumulative = cem43_cumulative(temps.as_slice().unwrap_or(&[]), dt_s);
    Ok(cumulative.last().copied().unwrap_or(0.0))
}

/// Compute the CEM43 rate at each temperature value.
///
/// For each temperature `T_i` in `temperatures_c`, returns:
///
/// ```text
/// result[i] = R(T_i)^{43 − T_i} · duration_s / 60
/// ```
///
/// This gives the CEM43 contribution of a single exposure of `duration_s`
/// seconds at that constant temperature — useful for CEM43(T) curves.
///
/// Parameters
/// ----------
/// temperatures_c : 1-D array of temperatures [°C].
/// duration_s : exposure duration for each temperature point [s].
///
/// Returns
/// -------
/// 1-D array of CEM43 values [min], same length as `temperatures_c`.
#[pyfunction]
pub fn cem43_at_temperatures<'py>(
    py: Python<'py>,
    temperatures_c: PyReadonlyArray1<f64>,
    duration_s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if duration_s < 0.0 {
        return Err(PyValueError::new_err("duration_s must be >= 0"));
    }
    let temps = temperatures_c.as_array();
    // Single constant exposure of `duration_s` at each temperature: the one-step
    // cumulative CEM43 from kwavers_physics (single source of truth for the formula).
    let result: Array1<f64> = temps.mapv(|t| {
        cem43_cumulative(&[t], duration_s)
            .first()
            .copied()
            .unwrap_or(0.0)
    });
    Ok(result.to_pyarray(py).into())
}

