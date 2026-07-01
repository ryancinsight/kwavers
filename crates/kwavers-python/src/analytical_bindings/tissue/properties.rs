//! Tissue property lookup bindings.

use kwavers_medium::absorption::{tissue_thermal_properties as thermal, AbsorptionTissueType as T};
use kwavers_physics::analytical::tissue;
use pyo3::prelude::*;

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
    Ok(thermal(absorption_tissue_type_by_name(&tissue)))
}

fn absorption_tissue_type_by_name(tissue: &str) -> T {
    match tissue.to_ascii_lowercase().as_str() {
        "liver" => T::Liver,
        "kidney" => T::Kidney,
        "brain" => T::Brain,
        "muscle" => T::Muscle,
        "fat" => T::Fat,
        "blood" => T::Blood,
        "water" => T::Water,
        _ => T::SoftTissue,
    }
}
