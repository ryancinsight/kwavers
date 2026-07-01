//! Regularization parameter-selection bindings.

use kwavers_math::inverse_problems::parameter_selection;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// L-curve corner index (point of maximum curvature of the log–log
/// residual-vs-model-norm trade-off, Hansen 1992; book §18.7). Returns the
/// interior index of the maximum-curvature sample, or `None` if the inputs are
/// too short (`n < 5`), mismatched, non-positive, or `lambdas` not increasing.
///
/// Args:
///     residual_norms: ‖F(m_λ)−d‖ over the λ sweep.
///     model_norms: ‖m_λ‖ over the λ sweep.
///     lambdas: Regularization weights (strictly increasing, > 0).
///
/// Returns:
///     Corner index (int) or None.
#[pyfunction]
#[pyo3(signature = (residual_norms, model_norms, lambdas))]
pub fn l_curve_corner(
    residual_norms: PyReadonlyArray1<f64>,
    model_norms: PyReadonlyArray1<f64>,
    lambdas: PyReadonlyArray1<f64>,
) -> PyResult<Option<usize>> {
    let r = residual_norms
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let m = model_norms
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let l = lambdas
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(parameter_selection::l_curve_corner(r, m, l))
}

/// Morozov discrepancy-principle regularization weight `λ*`: the `λ` at which the
/// data-residual norm equals `τ·δ` (noise level `δ`, safety factor `τ ≥ 1`),
/// found by linear interpolation (book §18.7). Returns `None` if `τ·δ` is outside
/// the sampled residual range or the inputs are invalid.
///
/// Args:
///     lambdas: Regularization weights over the sweep.
///     residual_norms: ‖F(m_λ)−d‖ over the sweep (monotone non-decreasing).
///     noise_level: Noise level δ.
///     tau: Safety factor τ ≥ 1.
///
/// Returns:
///     λ* (float) or None.
#[pyfunction]
#[pyo3(signature = (lambdas, residual_norms, noise_level, tau))]
pub fn morozov_lambda(
    lambdas: PyReadonlyArray1<f64>,
    residual_norms: PyReadonlyArray1<f64>,
    noise_level: f64,
    tau: f64,
) -> PyResult<Option<f64>> {
    let l = lambdas
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let r = residual_norms
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(parameter_selection::morozov_lambda(l, r, noise_level, tau))
}
