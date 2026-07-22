//! PMUT scalar model bindings.

use super::helpers::pmut;
use pyo3::prelude::*;

/// PMUT immersion resonance `Hz` (film = "aln" | "pzt").
#[pyfunction]
pub fn pmut_resonance_immersion(
    film: &str,
    radius: f64,
    t_p: f64,
    t_s: f64,
    density_fluid: f64,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.immersion_resonance(density_fluid))
}

/// PMUT effective electromechanical coupling k² [-].
#[pyfunction]
pub fn pmut_coupling_k2(film: &str, radius: f64, t_p: f64, t_s: f64) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.coupling_k2())
}

/// PMUT dielectric self-heating power `W`.
#[pyfunction]
pub fn pmut_self_heating(
    film: &str,
    radius: f64,
    t_p: f64,
    t_s: f64,
    v_ac: f64,
    freq: f64,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.self_heating_power(v_ac, freq))
}

/// PMUT fractional bandwidth from fluid loading [-].
#[pyfunction]
pub fn pmut_fractional_bandwidth(
    film: &str,
    radius: f64,
    t_p: f64,
    t_s: f64,
    density_fluid: f64,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.fractional_bandwidth(density_fluid))
}

/// PMUT drive-scaled peak output pressure `Pa` (film = "aln" | "pzt").
#[pyfunction]
pub fn pmut_max_output_pressure(
    film: &str,
    radius: f64,
    t_p: f64,
    t_s: f64,
    drive_voltage: f64,
    density_fluid: f64,
    sound_speed_fluid: f64,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.max_output_pressure(
        drive_voltage,
        density_fluid,
        sound_speed_fluid,
    ))
}
