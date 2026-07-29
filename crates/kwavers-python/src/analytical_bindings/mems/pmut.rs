//! PMUT scalar model bindings.

use aequitas::systems::si::quantities::{ElectricPotential, Frequency, MassDensity, Velocity};
use super::py_convert::pmut;
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
    Ok(pmut(film, radius, t_p, t_s)?.immersion_resonance(MassDensity::from_base(density_fluid)).into_base())
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
    Ok(pmut(film, radius, t_p, t_s)?
        .self_heating_power(ElectricPotential::from_base(v_ac), Frequency::from_base(freq))
        .into_base())
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
    Ok(pmut(film, radius, t_p, t_s)?.fractional_bandwidth(MassDensity::from_base(density_fluid)))
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
    Ok(pmut(film, radius, t_p, t_s)?
        .max_output_pressure(
            ElectricPotential::from_base(drive_voltage),
            MassDensity::from_base(density_fluid),
            Velocity::from_base(sound_speed_fluid),
        )
        .into_base())
}

