//! CMUT scalar model bindings.

use super::helpers::cmut;
use aequitas::systems::si::units::{Hertz, Pascal, Volt, Watt};
use pyo3::prelude::*;

use crate::quantity_args::{
    PyDimensionless, PyElectricPotential, PyFrequency, PyLength, PyMassDensity, PyReciprocalLength,
    PyVelocity,
};

/// CMUT (Si) immersion resonance `Hz`.
#[pyfunction]
pub fn cmut_resonance_immersion(
    radius: PyLength,
    thickness: PyLength,
    gap: PyLength,
    density_fluid: PyMassDensity,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?
        .immersion_resonance(density_fluid.quantity())
        .in_unit::<Hertz>())
}

/// CMUT collapse (pull-in) voltage `V`.
#[pyfunction]
pub fn cmut_collapse_voltage(
    radius: PyLength,
    thickness: PyLength,
    gap: PyLength,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?
        .collapse_voltage()
        .in_unit::<Volt>())
}

/// CMUT bias-dependent electromechanical coupling k² [-].
#[pyfunction]
pub fn cmut_coupling_k2(
    radius: PyLength,
    thickness: PyLength,
    gap: PyLength,
    bias_voltage: PyElectricPotential,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?
        .coupling_k2(bias_voltage.quantity())
        .into_base())
}

/// CMUT dielectric self-heating power `W`.
#[pyfunction]
pub fn cmut_self_heating(
    radius: PyLength,
    thickness: PyLength,
    gap: PyLength,
    v_ac: PyElectricPotential,
    freq: PyFrequency,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?
        .self_heating_power(v_ac.quantity(), freq.quantity())
        .in_unit::<Watt>())
}

/// CMUT fractional bandwidth from fluid loading [-].
#[pyfunction]
pub fn cmut_fractional_bandwidth(
    radius: PyLength,
    thickness: PyLength,
    density_fluid: PyMassDensity,
) -> PyResult<f64> {
    // gap does not affect bandwidth; use a nominal value for construction
    Ok(cmut(radius, thickness, PyLength::from_base(0.1e-6))?
        .fractional_bandwidth(density_fluid.quantity())
        .into_base())
}

/// CMUT gap-limited peak output pressure `Pa` (swing_fraction ≈ 1/3 conventional).
#[pyfunction]
pub fn cmut_max_output_pressure(
    radius: PyLength,
    thickness: PyLength,
    gap: PyLength,
    density_fluid: PyMassDensity,
    sound_speed_fluid: PyVelocity,
    swing_fraction: PyDimensionless,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?
        .max_output_pressure(
            density_fluid.quantity(),
            sound_speed_fluid.quantity(),
            swing_fraction.quantity(),
        )
        .in_unit::<Pascal>())
}

/// CMUT output derating when flexed to curvature `curvature` [1/m].
#[pyfunction]
pub fn cmut_flex_gap_derating(
    radius: PyLength,
    thickness: PyLength,
    gap: PyLength,
    curvature: PyReciprocalLength,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?
        .flex_gap_derating(curvature.quantity())
        .into_base())
}
