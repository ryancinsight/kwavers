mod slice_fwi;
mod volume_fwi;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub(crate) fn kwavers_to_py(err: kwavers::core::error::KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers seismic FWI failed: {err}"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        slice_fwi::run_seismic_helmet_fwi_from_ritk_ct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        volume_fwi::run_seismic_helmet_fwi_volume_from_ritk_ct,
        m
    )?)?;
    Ok(())
}
