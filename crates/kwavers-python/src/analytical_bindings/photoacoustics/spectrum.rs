//! Spectral absorption and Gruneisen photoacoustic bindings.

use kwavers_physics::analytical::photoacoustics;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Return the molar absorption spectrum of oxyhaemoglobin (HbO2).
///
/// Args:
///     wavelength_nm: Wavelength array [nm].
///
/// Returns:
///     Molar absorption coefficient array [cm⁻¹/M].
#[pyfunction]
#[pyo3(signature = (wavelength_nm,))]
pub fn hbo2_molar_absorption(
    py: Python<'_>,
    wavelength_nm: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let w_s = wavelength_nm
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::hbo2_molar_absorption(w_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Return the molar absorption spectrum of deoxyhaemoglobin (Hb).
///
/// Args:
///     wavelength_nm: Wavelength array [nm].
///
/// Returns:
///     Molar absorption coefficient array [cm⁻¹/M].
#[pyfunction]
#[pyo3(signature = (wavelength_nm,))]
pub fn hb_molar_absorption(
    py: Python<'_>,
    wavelength_nm: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let w_s = wavelength_nm
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::hb_molar_absorption(w_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the Grüneisen parameter of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Grüneisen parameter array (dimensionless).
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
pub fn gruneisen_parameter_water(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::gruneisen_parameter_water(t_s);
    Ok(result.to_pyarray(py).unbind())
}

/// Compute the Grüneisen parameter of generic soft tissue vs temperature
/// (linear PA-thermometry model, Xu & Wang 2006).
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Grüneisen parameter array (dimensionless).
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
pub fn gruneisen_parameter_soft_tissue(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::gruneisen_parameter_soft_tissue(t_s);
    Ok(result.to_pyarray(py).unbind())
}

