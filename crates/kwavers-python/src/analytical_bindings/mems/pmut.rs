//! PMUT scalar model bindings.

use super::helpers::pmut;
use aequitas::systems::si::units::{Hertz, Pascal, Watt};
use pyo3::prelude::*;

use crate::quantity_args::{PyElectricPotential, PyFrequency, PyLength, PyMassDensity, PyVelocity};

/// PMUT immersion resonance `Hz` (film = "aln" | "pzt").
#[pyfunction]
pub fn pmut_resonance_immersion(
    film: &str,
    radius: PyLength,
    t_p: PyLength,
    t_s: PyLength,
    density_fluid: PyMassDensity,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?
        .immersion_resonance(density_fluid.quantity())
        .in_unit::<Hertz>())
}

/// PMUT effective electromechanical coupling k² [-].
#[pyfunction]
pub fn pmut_coupling_k2(
    film: &str,
    radius: PyLength,
    t_p: PyLength,
    t_s: PyLength,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.coupling_k2().into_base())
}

/// PMUT dielectric self-heating power `W`.
#[pyfunction]
pub fn pmut_self_heating(
    film: &str,
    radius: PyLength,
    t_p: PyLength,
    t_s: PyLength,
    v_ac: PyElectricPotential,
    freq: PyFrequency,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?
        .self_heating_power(v_ac.quantity(), freq.quantity())
        .in_unit::<Watt>())
}

/// PMUT fractional bandwidth from fluid loading [-].
#[pyfunction]
pub fn pmut_fractional_bandwidth(
    film: &str,
    radius: PyLength,
    t_p: PyLength,
    t_s: PyLength,
    density_fluid: PyMassDensity,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?
        .fractional_bandwidth(density_fluid.quantity())
        .into_base())
}

/// PMUT drive-scaled peak output pressure `Pa` (film = "aln" | "pzt").
#[pyfunction]
pub fn pmut_max_output_pressure(
    film: &str,
    radius: PyLength,
    t_p: PyLength,
    t_s: PyLength,
    drive_voltage: PyElectricPotential,
    density_fluid: PyMassDensity,
    sound_speed_fluid: PyVelocity,
) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?
        .max_output_pressure(
            drive_voltage.quantity(),
            density_fluid.quantity(),
            sound_speed_fluid.quantity(),
        )
        .in_unit::<Pascal>())
}
