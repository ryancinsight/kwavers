use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (db, y=1.0))]
fn db2neper(db: f64, y: f64) -> f64 {
    let neper_per_db = 10.0f64.ln() / 20.0;
    db * (100.0 * neper_per_db) / (2.0 * std::f64::consts::PI * 1e6).powf(y)
}

#[pyfunction]
#[pyo3(signature = (neper, y=1.0))]
fn neper2db(neper: f64, y: f64) -> f64 {
    let db_per_neper = 20.0 / 10.0f64.ln();
    neper * (db_per_neper / 100.0) * (2.0 * std::f64::consts::PI * 1e6).powf(y)
}

#[pyfunction]
fn freq2wavenumber(frequency: f64, sound_speed: f64) -> PyResult<f64> {
    if sound_speed <= 0.0 {
        return Err(PyValueError::new_err("Sound speed must be positive"));
    }
    if frequency < 0.0 {
        return Err(PyValueError::new_err("Frequency must be non-negative"));
    }
    Ok(2.0 * std::f64::consts::PI * frequency / sound_speed)
}

#[pyfunction]
fn hounsfield2density(hu: f64) -> f64 {
    kwavers::core::constants::hounsfield::HounsfieldUnits::to_density(hu)
}

#[pyfunction]
fn hounsfield2soundspeed(hu: f64) -> f64 {
    kwavers::core::constants::hounsfield::HounsfieldUnits::to_sound_speed(hu)
}

pub(super) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(db2neper, m)?)?;
    m.add_function(wrap_pyfunction!(neper2db, m)?)?;
    m.add_function(wrap_pyfunction!(freq2wavenumber, m)?)?;
    m.add_function(wrap_pyfunction!(hounsfield2density, m)?)?;
    m.add_function(wrap_pyfunction!(hounsfield2soundspeed, m)?)?;
    Ok(())
}
