//! PyO3 wrappers for transducer interpolation helpers.

use kwavers_physics::analytical::transducer;
use ndarray::Array2;
use numpy::{ToPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Compute band-limited interpolation (BLI) stencil weights.
///
/// Args:
///     delta: Sub-sample offsets (1-D array).
///     n_stencil: Number of stencil points.
///
/// Returns:
///     2-D array of shape (len(delta), n_stencil).
#[pyfunction]
#[pyo3(signature = (delta, n_stencil))]
pub fn bli_stencil_weights(
    py: Python<'_>,
    delta: PyReadonlyArray1<f64>,
    n_stencil: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let d_s = delta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rows: Vec<Vec<f64>> = transducer::bli_stencil_weights(d_s, n_stencil);
    let n_delta = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr2d = Array2::from_shape_vec((n_delta, n_stencil), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.to_pyarray(py).unbind())
}

/// Compute nearest-neighbour and BLI RMS interpolation-error curves.
///
/// Args:
///     ppw: Points per wavelength samples.
///     delta: Fractional grid offsets in sample units.
///     n_stencil: Even BLI stencil length.
///
/// Returns:
///     (nearest_rms, bli_rms), both length len(ppw).
#[pyfunction]
#[pyo3(signature = (ppw, delta, n_stencil))]
pub fn bli_interpolation_error_curves(
    py: Python<'_>,
    ppw: PyReadonlyArray1<f64>,
    delta: PyReadonlyArray1<f64>,
    n_stencil: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let ppw_s = ppw
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let delta_s = delta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if ppw_s.is_empty() {
        return Err(PyValueError::new_err("ppw must not be empty"));
    }
    if delta_s.is_empty() {
        return Err(PyValueError::new_err("delta must not be empty"));
    }
    if n_stencil == 0 || !n_stencil.is_multiple_of(2) {
        return Err(PyValueError::new_err(
            "n_stencil must be a positive even integer",
        ));
    }
    if ppw_s.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(PyValueError::new_err(
            "ppw entries must be finite and positive",
        ));
    }
    if delta_s.iter().any(|&v| !v.is_finite()) {
        return Err(PyValueError::new_err("delta entries must be finite"));
    }
    let (nearest, bli) = transducer::bli_interpolation_error_curves(ppw_s, delta_s, n_stencil);
    Ok((
        nearest.to_pyarray(py).unbind(),
        bli.to_pyarray(py).unbind(),
    ))
}

