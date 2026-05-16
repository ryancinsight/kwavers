mod helpers;
mod sources;

pub(crate) use sources::process_source_for_run;

use kwavers::core::error::KwaversError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Convert a [`KwaversError`] to a [`PyErr`] (PyRuntimeError).
pub(crate) fn kwavers_error_to_py_local(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers error: {}", err))
}
