//! NumPy conversion helpers for inverse-problem bindings.

use crate::array_utils::{copy_pyarray2_to_vec, vec_to_pyarray2};
use numpy::{PyArray2, PyReadonlyArray2};

use pyo3::prelude::*;

pub(super) fn flatten_array2(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f64>> {
    let (data, _) = copy_pyarray2_to_vec(arr)?;
    Ok(data)
}

pub(super) fn array2_from_flat(
    py: Python<'_>,
    nrows: usize,
    ncols: usize,
    flat: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    vec_to_pyarray2(py, [nrows, ncols], flat)
}
