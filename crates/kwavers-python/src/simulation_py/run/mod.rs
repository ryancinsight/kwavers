mod execute;
mod helpers;
mod prepare;
pub(crate) mod sources;

use kwavers_core::error::KwaversError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Convert a `KwaversError` to a [`PyErr`] (PyRuntimeError).
pub(crate) fn kwavers_error_to_py(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers error: {}", err))
}
