//! CMUT scalar model bindings.

use super::helpers::cmut;
use pyo3::prelude::*;

/// CMUT (Si) immersion resonance [Hz].
#[pyfunction]
pub fn cmut_resonance_immersion(
    radius: f64,
    thickness: f64,
    gap: f64,
    density_fluid: f64,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.immersion_resonance(density_fluid))
}

/// CMUT collapse (pull-in) voltage [V].
#[pyfunction]
pub fn cmut_collapse_voltage(radius: f64, thickness: f64, gap: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.collapse_voltage())
}

/// CMUT bias-dependent electromechanical coupling k² [-].
#[pyfunction]
pub fn cmut_coupling_k2(radius: f64, thickness: f64, gap: f64, bias_voltage: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.coupling_k2(bias_voltage))
}

/// CMUT dielectric self-heating power [W].
#[pyfunction]
pub fn cmut_self_heating(
    radius: f64,
    thickness: f64,
    gap: f64,
    v_ac: f64,
    freq: f64,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.self_heating_power(v_ac, freq))
}

/// CMUT fractional bandwidth from fluid loading [-].
#[pyfunction]
pub fn cmut_fractional_bandwidth(radius: f64, thickness: f64, density_fluid: f64) -> PyResult<f64> {
    // gap does not affect bandwidth; use a nominal value for construction
    Ok(cmut(radius, thickness, 0.1e-6)?.fractional_bandwidth(density_fluid))
}

/// CMUT gap-limited peak output pressure [Pa] (swing_fraction ≈ 1/3 conventional).
#[pyfunction]
pub fn cmut_max_output_pressure(
    radius: f64,
    thickness: f64,
    gap: f64,
    density_fluid: f64,
    sound_speed_fluid: f64,
    swing_fraction: f64,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.max_output_pressure(
        density_fluid,
        sound_speed_fluid,
        swing_fraction,
    ))
}

/// CMUT output derating when flexed to curvature `curvature` [1/m].
#[pyfunction]
pub fn cmut_flex_gap_derating(
    radius: f64,
    thickness: f64,
    gap: f64,
    curvature: f64,
) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.flex_gap_derating(curvature))
}
