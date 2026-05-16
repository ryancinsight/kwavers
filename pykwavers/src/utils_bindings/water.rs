use pyo3::prelude::*;

#[pyfunction]
fn water_sound_speed(temp_celsius: f64) -> f64 {
    kwavers::core::constants::water::WaterProperties::sound_speed(temp_celsius)
}

#[pyfunction]
fn water_density(temp_celsius: f64) -> f64 {
    kwavers::core::constants::water::WaterProperties::density(temp_celsius)
}

#[pyfunction]
fn water_absorption(frequency: f64, temp_celsius: f64) -> f64 {
    let freq_mhz = frequency / 1e6;
    let db_per_cm = kwavers::core::constants::water::WaterProperties::absorption_pinkerton(
        freq_mhz,
        temp_celsius,
    );
    db_per_cm / 8.686 * 100.0
}

#[pyfunction]
fn water_nonlinearity(temp_celsius: f64) -> f64 {
    kwavers::core::constants::water::WaterProperties::nonlinear_parameter(temp_celsius)
}

pub(super) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(water_sound_speed, m)?)?;
    m.add_function(wrap_pyfunction!(water_density, m)?)?;
    m.add_function(wrap_pyfunction!(water_absorption, m)?)?;
    m.add_function(wrap_pyfunction!(water_nonlinearity, m)?)?;
    Ok(())
}
