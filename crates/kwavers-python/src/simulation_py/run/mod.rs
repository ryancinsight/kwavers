mod helpers;
pub(crate) mod sources;

use kwavers_core::error::KwaversError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Convert a `KwaversError` to a [`PyErr`] (PyRuntimeError).
pub(crate) fn kwavers_error_to_py(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers error: {}", err))
}

/// Legacy alias kept for old solver files that still reference this name.
pub(crate) fn kwavers_error_to_py_local(err: KwaversError) -> PyErr {
    kwavers_error_to_py(err)
}