mod transcranial_slice_inversion;
mod transcranial_volume_inversion;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub(crate) fn kwavers_to_py(err: kwavers_core::error::KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers transcranial UST inversion failed: {err}"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        transcranial_slice_inversion::run_transcranial_ust_slice_inversion_from_ritk_ct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        transcranial_volume_inversion::run_transcranial_ust_volume_inversion_from_ritk_ct,
        m
    )?)?;
    Ok(())
}
