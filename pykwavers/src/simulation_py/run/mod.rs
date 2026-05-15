mod sources;
mod helpers;

pub(crate) use sources::process_source_for_run;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use kwavers::core::error::KwaversError;

/// Convert a [`KwaversError`] to a [`PyErr`] (PyRuntimeError).
pub(crate) fn kwavers_error_to_py_local(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers error: {}", err))
}
