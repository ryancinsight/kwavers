//! NumPy array conversion helpers for statistics bindings.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(super) fn as_slices<'a>(
    a: &'a PyReadonlyArray1<f64>,
    b: &'a PyReadonlyArray1<f64>,
) -> PyResult<(&'a [f64], &'a [f64])> {
    let a_s = a
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let b_s = b
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((a_s, b_s))
}
