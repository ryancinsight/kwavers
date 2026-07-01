//! NumPy conversion helpers for RTM bindings.

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(super) fn flatten_array2(arr: ArrayView2<'_, f64>, nx: usize, nz: usize) -> Vec<f64> {
    (0..nx)
        .flat_map(|i| (0..nz).map(move |j| arr[[i, j]]))
        .collect()
}

pub(super) fn array2_from_flat(
    py: Python<'_>,
    nx: usize,
    nz: usize,
    flat: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr2d = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

pub(super) fn complex_field_arrays(
    py: Python<'_>,
    nx: usize,
    nz: usize,
    real_flat: Vec<f64>,
    imag_flat: Vec<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    Ok((
        array2_from_flat(py, nx, nz, real_flat)?,
        array2_from_flat(py, nx, nz, imag_flat)?,
    ))
}
