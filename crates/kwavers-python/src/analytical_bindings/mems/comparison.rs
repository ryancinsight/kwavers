//! CMUT/PMUT comparison figure-of-merit bindings.

use super::helpers::{cmut, pmut};
use kwavers_transducer::mems::comparison;
use pyo3::prelude::*;

/// Therapy comparison. Returns
/// `[cmut_output_pa, pmut_output_pa, cmut_flex_derating, cmut_heating,
/// pmut_heating, recommended]` (recommended 0=CMUT, 1=PMUT).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn therapy_figure_of_merit(
    cmut_radius: f64,
    cmut_thickness: f64,
    cmut_gap: f64,
    pmut_film: &str,
    pmut_radius: f64,
    pmut_t_p: f64,
    pmut_t_s: f64,
    fluid_density: f64,
    fluid_sound_speed: f64,
    cmut_swing_fraction: f64,
    pmut_drive_voltage: f64,
    curvature: f64,
    substrate_output_factor: f64,
) -> PyResult<Vec<f64>> {
    let c = cmut(cmut_radius, cmut_thickness, cmut_gap)?;
    let p = pmut(pmut_film, pmut_radius, pmut_t_p, pmut_t_s)?;
    let v = comparison::evaluate_therapy(
        &c,
        &p,
        fluid_density,
        fluid_sound_speed,
        cmut_swing_fraction,
        pmut_drive_voltage,
        curvature,
        substrate_output_factor,
    );
    let recommended = if v.recommended == comparison::MutKind::Cmut {
        0.0
    } else {
        1.0
    };
    Ok(vec![
        v.cmut_output_pa,
        v.pmut_output_pa,
        v.cmut_flex_derating,
        v.cmut_heating,
        v.pmut_heating,
        recommended,
    ])
}

/// IVUS figure-of-merit comparison.
///
/// Returns `[cmut_fbw, pmut_fbw, cmut_heating, pmut_heating, cmut_drive_v,
/// pmut_drive_v, cmut_fom, pmut_fom, recommended]` where `recommended` is
/// `0.0` for CMUT and `1.0` for PMUT.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ivus_figure_of_merit(
    cmut_radius: f64,
    cmut_thickness: f64,
    cmut_gap: f64,
    pmut_film: &str,
    pmut_radius: f64,
    pmut_t_p: f64,
    pmut_t_s: f64,
    fluid_density: f64,
    pmut_drive_voltage: f64,
) -> PyResult<Vec<f64>> {
    let c = cmut(cmut_radius, cmut_thickness, cmut_gap)?;
    let p = pmut(pmut_film, pmut_radius, pmut_t_p, pmut_t_s)?;
    let v = comparison::evaluate_ivus(
        &c,
        &p,
        fluid_density,
        pmut_drive_voltage,
        comparison::IvusWeights::default(),
    );
    let recommended = if v.recommended == comparison::MutKind::Cmut {
        0.0
    } else {
        1.0
    };
    Ok(vec![
        v.cmut_fbw,
        v.pmut_fbw,
        v.cmut_heating,
        v.pmut_heating,
        v.cmut_drive_voltage,
        v.pmut_drive_voltage,
        v.cmut_fom,
        v.pmut_fom,
        recommended,
    ])
}
