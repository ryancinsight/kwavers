use pyo3::prelude::*;

#[pyfunction]
fn water_sound_speed(temp_celsius: f64) -> f64 {
    kwavers_core::constants::water::WaterProperties::sound_speed(temp_celsius)
}

#[pyfunction]
fn water_density(temp_celsius: f64) -> f64 {
    kwavers_core::constants::water::WaterProperties::density(temp_celsius)
}

#[pyfunction]
fn water_absorption(frequency: f64, temp_celsius: f64) -> f64 {
    kwavers_core::constants::water::WaterProperties::absorption_pinkerton_np_per_m(
        frequency,
        temp_celsius,
    )
}

#[pyfunction]
fn water_nonlinearity(temp_celsius: f64) -> f64 {
    kwavers_core::constants::water::WaterProperties::nonlinear_parameter(temp_celsius)
}

pub(super) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(water_sound_speed, m)?)?;
    m.add_function(wrap_pyfunction!(water_density, m)?)?;
    m.add_function(wrap_pyfunction!(water_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(water_nonlinearity, m)?)?;
    Ok(())
}
