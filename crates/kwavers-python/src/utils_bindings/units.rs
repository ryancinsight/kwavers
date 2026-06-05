use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (db, y=1.0))]
fn db2neper(db: f64, y: f64) -> f64 {
    kwavers_core::units::db_per_mhz_cm_to_neper_per_rad_s_m(db, y)
}

#[pyfunction]
#[pyo3(signature = (neper, y=1.0))]
fn neper2db(neper: f64, y: f64) -> f64 {
    kwavers_core::units::neper_per_rad_s_m_to_db_per_mhz_cm(neper, y)
}

#[pyfunction]
fn freq2wavenumber(frequency: f64, sound_speed: f64) -> PyResult<f64> {
    kwavers_core::units::frequency_to_wavenumber(frequency, sound_speed)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
fn hounsfield2density(hu: f64) -> f64 {
    kwavers_core::constants::hounsfield::HounsfieldUnits::to_density(hu)
}

#[pyfunction]
fn hounsfield2soundspeed(hu: f64) -> f64 {
    kwavers_core::constants::hounsfield::HounsfieldUnits::to_sound_speed(hu)
}

pub(super) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(db2neper, m)?)?;
    m.add_function(wrap_pyfunction!(neper2db, m)?)?;
    m.add_function(wrap_pyfunction!(freq2wavenumber, m)?)?;
    m.add_function(wrap_pyfunction!(hounsfield2density, m)?)?;
    m.add_function(wrap_pyfunction!(hounsfield2soundspeed, m)?)?;
    Ok(())
}
