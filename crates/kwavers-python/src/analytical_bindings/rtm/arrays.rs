//! NumPy conversion helpers for RTM bindings.

use crate::array_utils::{copy_pyarray2_to_vec, vec_to_pyarray2};
use numpy::{PyArray2, PyReadonlyArray2};

use pyo3::prelude::*;

pub(super) fn flatten_array2(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f64>> {
    let (data, _) = copy_pyarray2_to_vec(arr)?;
    Ok(data)
}

pub(super) fn array2_from_flat(
    py: Python<'_>,
    nx: usize,
    nz: usize,
    flat: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    vec_to_pyarray2(py, [nx, nz], flat)
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
