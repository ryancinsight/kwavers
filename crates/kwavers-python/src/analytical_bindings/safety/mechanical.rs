//! Mechanical-index and cavitation-risk safety bindings.

use kwavers_physics::analytical::safety;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the Mechanical Index (MI).
///
/// MI = |p_neg_pa| / (1e6 * sqrt(f_hz / 1e6))
///
/// Args:
///     p_neg_pa: Peak negative pressure `Pa`.
///     f_hz: Frequency `Hz`.
///
/// Returns:
///     Mechanical Index (dimensionless).
#[pyfunction]
#[pyo3(signature = (p_neg_pa, f_hz))]
pub fn mechanical_index(p_neg_pa: f64, f_hz: f64) -> PyResult<f64> {
    Ok(safety::mechanical_index(p_neg_pa, f_hz))
}

/// Compute the Mechanical Index over a pressure field (array variant).
///
/// Applies MI = |p_field`i`| / (1e6 * sqrt(f_hz / 1e6)) element-wise.
///
/// Args:
///     p_field: Peak rarefactional pressure field `Pa`, passed as 1-D array.
///     f_hz: Centre frequency `Hz`.
///
/// Returns:
///     MI array, same length as p_field (dimensionless).
///
/// Reference:
///     FDA Marketing Clearance of Diagnostic Ultrasound Systems, Appendix A.
#[pyfunction]
#[pyo3(signature = (p_field, f_hz))]
pub fn mechanical_index_field(
    py: Python<'_>,
    p_field: PyReadonlyArray1<f64>,
    f_hz: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = p_field
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::mechanical_index_field(p_s, f_hz);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the Mechanical Index for one pressure over a frequency sweep.
///
/// Args:
///     p_neg_pa: Peak rarefactional pressure threshold `Pa`.
///     f_hz: Centre frequencies `Hz`, passed as a 1-D array.
///
/// Returns:
///     MI array, same length as `f_hz`.
#[pyfunction]
#[pyo3(signature = (p_neg_pa, f_hz))]
pub fn mechanical_index_frequency_sweep(
    py: Python<'_>,
    p_neg_pa: f64,
    f_hz: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::mechanical_index_frequency_sweep(p_neg_pa, f_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute cavitation-risk probability from Mechanical Index.
///
/// Applies P_risk(MI) = 1 / (1 + exp[-s * (MI - MI_thr)]) element-wise.
///
/// Args:
///     mechanical_index: Mechanical Index samples [-], passed as 1-D array.
///     threshold_mi: MI at 50% cavitation risk.
///     slope: Logistic slope in reciprocal MI units.
///
/// Returns:
///     Cavitation-risk probability array, same length as mechanical_index.
#[pyfunction]
#[pyo3(signature = (mechanical_index, threshold_mi, slope))]
pub fn mechanical_index_cavitation_risk(
    py: Python<'_>,
    mechanical_index: PyReadonlyArray1<f64>,
    threshold_mi: f64,
    slope: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let mi_s = mechanical_index
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::mechanical_index_cavitation_risk(mi_s, threshold_mi, slope);
    Ok(result.to_pyarray(py).unbind())
}
