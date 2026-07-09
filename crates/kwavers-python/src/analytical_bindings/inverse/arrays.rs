//! NumPy conversion helpers for inverse-problem bindings.

use leto::{
    Array2,
    ArrayView2,
};
use numpy::{ToPyArray, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(super) fn flatten_array2(arr: ArrayView2<'_, f64>, nrows: usize, ncols: usize) -> Vec<f64> {
    (0..nrows)
        .flat_map(|i| (0..ncols).map(move |j| arr[[i, j]]))
        .collect()
}

pub(super) fn array2_from_flat(
    py: Python<'_>,
    nrows: usize,
    ncols: usize,
    flat: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr2d = Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.to_pyarray(py).unbind())
}

