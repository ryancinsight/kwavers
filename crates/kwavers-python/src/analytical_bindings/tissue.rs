//! PyO3 bindings for `kwavers_physics::analytical::tissue`.

use kwavers_physics::analytical::tissue;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the sound speed of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
pub fn water_sound_speed_temperature(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::water_sound_speed_temperature(t_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the density of water as a function of temperature.
///
/// Args:
///     t_celsius: Temperature array [°C].
///
/// Returns:
///     Density array [kg/m³].
#[pyfunction]
#[pyo3(signature = (t_celsius,))]
pub fn water_density_temperature(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::water_density_temperature(t_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Return the B/A nonlinearity parameter for a named medium.
///
/// Supported values: "water", "blood", "fat", "liver", "kidney", "brain",
/// "muscle", "bone".
///
/// Args:
///     medium: Medium name string.
///
/// Returns:
///     B/A value.
#[pyfunction]
#[pyo3(signature = (medium,))]
pub fn ba_parameter(medium: String) -> PyResult<f64> {
    Ok(tissue::ba_parameter(&medium))
}

/// Compute frequency-dependent tissue absorption in dB/cm.
///
/// Args:
///     f_mhz: Frequency array [MHz].
///     tissue: Tissue name string.
///
/// Returns:
///     Absorption array [dB/cm].
#[pyfunction]
#[pyo3(signature = (f_mhz, tissue))]
pub fn tissue_absorption_db_cm(
    py: Python<'_>,
    f_mhz: PyReadonlyArray1<f64>,
    tissue: String,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_mhz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::tissue_absorption_db_cm(f_s, &tissue);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Kramers–Kronig consistent sound speed dispersion.
///
/// Args:
///     f_hz: Frequency array [Hz].
///     alpha0: Attenuation coefficient [Np/m/Hz^y].
///     y: Power-law exponent.
///     f_ref_hz: Reference frequency [Hz].
///     c_ref: Sound speed at *f_ref_hz* [m/s].
///
/// Returns:
///     Sound speed array [m/s].
#[pyfunction]
#[pyo3(signature = (f_hz, alpha0, y, f_ref_hz, c_ref))]
pub fn kramers_kronig_sound_speed(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    alpha0: f64,
    y: f64,
    f_ref_hz: f64,
    c_ref: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = tissue::kramers_kronig_sound_speed(f_s, alpha0, y, f_ref_hz, c_ref);
    Ok(result.into_pyarray(py).unbind())
}

/// Return tabulated acoustic properties for a named tissue.
///
/// Args:
///     tissue: Tissue name string.
///
/// Returns:
///     (sound_speed_m_s, density_kg_m3, attenuation_db_cm_mhz,
///      nonlinearity_ba, impedance_mrayl) — all f64.
#[pyfunction]
#[pyo3(signature = (tissue,))]
pub fn tissue_properties(tissue: String) -> PyResult<(f64, f64, f64, f64, f64)> {
    Ok(tissue::tissue_properties(&tissue))
}

/// Histotripsy mechanical / cavitation-threshold tissue characterization, from
/// the kwavers-domain tissue database (Maxwell 2013; Vlaisavljevich 2014/2015).
///
/// Args:
///     tissue: Tissue name string (e.g. "liver", "kidney", "brain").
///
/// Returns:
///     (tensile_yield_stress_pa, intrinsic_threshold_1mhz_pa,
///      threshold_slope_pa_per_decade, threshold_sigma_pa) — all f64.
#[pyfunction]
#[pyo3(signature = (tissue,))]
pub fn histotripsy_tissue_properties(tissue: String) -> PyResult<(f64, f64, f64, f64)> {
    let p = kwavers_medium::absorption::histotripsy_tissue_properties_by_name(&tissue);
    Ok((
        p.tensile_yield_stress_pa,
        p.intrinsic_threshold_1mhz_pa,
        p.threshold_slope_pa_per_decade,
        p.threshold_sigma_pa,
    ))
}

/// Thermal/acoustic tissue properties from the kwavers-domain tissue database,
/// for shock-heating / boiling-histotripsy models.
///
/// Args:
///     tissue: Tissue name string.
///
/// Returns:
///     (specific_heat_J_per_kgK, thermal_conductivity_W_per_mK, density_kg_m3).
#[pyfunction]
#[pyo3(signature = (tissue,))]
pub fn tissue_thermal_properties(tissue: String) -> PyResult<(f64, f64, f64)> {
    use kwavers_medium::absorption::{
        tissue_thermal_properties as thermal, AbsorptionTissueType as T,
    };
    let t = match tissue.to_ascii_lowercase().as_str() {
        "liver" => T::Liver,
        "kidney" => T::Kidney,
        "brain" => T::Brain,
        "muscle" => T::Muscle,
        "fat" => T::Fat,
        "blood" => T::Blood,
        "water" => T::Water,
        _ => T::SoftTissue,
    };
    Ok(thermal(t))
}
