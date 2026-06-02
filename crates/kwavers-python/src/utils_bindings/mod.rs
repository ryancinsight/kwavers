mod geometry;
mod signal;
mod units;
mod water;

use pyo3::prelude::*;

pub fn register_utils(m: &Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    signal::register(m)?;
    geometry::register(m)?;
    units::register(m)?;
    water::register(m)?;
    Ok(())
}
