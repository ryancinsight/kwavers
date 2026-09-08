//! CMUT/PMUT comparison figure-of-merit bindings.

use super::helpers::{cmut, pmut};
use aequitas::systems::si::units::{Pascal, Volt, Watt};
use kwavers_transducer::mems::comparison;
use pyo3::prelude::*;

use crate::quantity_args::{
    PyDimensionless, PyElectricPotential, PyLength, PyMassDensity, PyReciprocalLength, PyVelocity,
};

/// Therapy comparison. Returns
/// `[cmut_output_pa, pmut_output_pa, cmut_flex_derating, cmut_heating,
/// pmut_heating, recommended]` (recommended 0=CMUT, 1=PMUT).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn therapy_figure_of_merit(
    cmut_radius: PyLength,
    cmut_thickness: PyLength,
    cmut_gap: PyLength,
    pmut_film: &str,
    pmut_radius: PyLength,
    pmut_t_p: PyLength,
    pmut_t_s: PyLength,
    fluid_density: PyMassDensity,
    fluid_sound_speed: PyVelocity,
    cmut_swing_fraction: PyDimensionless,
    pmut_drive_voltage: PyElectricPotential,
    curvature: PyReciprocalLength,
    substrate_output_factor: PyDimensionless,
) -> PyResult<Vec<f64>> {
    let c = cmut(cmut_radius, cmut_thickness, cmut_gap)?;
    let p = pmut(pmut_film, pmut_radius, pmut_t_p, pmut_t_s)?;
    let v = comparison::evaluate_therapy(
        &c,
        &p,
        fluid_density.quantity(),
        fluid_sound_speed.quantity(),
        cmut_swing_fraction.quantity(),
        pmut_drive_voltage.quantity(),
        curvature.quantity(),
        substrate_output_factor.quantity(),
    );
    let recommended = if v.recommended == comparison::MutKind::Cmut {
        0.0
    } else {
        1.0
    };
    Ok(vec![
        v.cmut_output_pa.in_unit::<Pascal>(),
        v.pmut_output_pa.in_unit::<Pascal>(),
        v.cmut_flex_derating.into_base(),
        v.cmut_heating.in_unit::<Watt>(),
        v.pmut_heating.in_unit::<Watt>(),
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
    cmut_radius: PyLength,
    cmut_thickness: PyLength,
    cmut_gap: PyLength,
    pmut_film: &str,
    pmut_radius: PyLength,
    pmut_t_p: PyLength,
    pmut_t_s: PyLength,
    fluid_density: PyMassDensity,
    pmut_drive_voltage: PyElectricPotential,
) -> PyResult<Vec<f64>> {
    let c = cmut(cmut_radius, cmut_thickness, cmut_gap)?;
    let p = pmut(pmut_film, pmut_radius, pmut_t_p, pmut_t_s)?;
    let v = comparison::evaluate_ivus(
        &c,
        &p,
        fluid_density.quantity(),
        pmut_drive_voltage.quantity(),
        comparison::IvusWeights::default(),
    );
    let recommended = if v.recommended == comparison::MutKind::Cmut {
        0.0
    } else {
        1.0
    };
    Ok(vec![
        v.cmut_fbw.into_base(),
        v.pmut_fbw.into_base(),
        v.cmut_heating.in_unit::<Watt>(),
        v.pmut_heating.in_unit::<Watt>(),
        v.cmut_drive_voltage.in_unit::<Volt>(),
        v.pmut_drive_voltage.in_unit::<Volt>(),
        v.cmut_fom.into_base(),
        v.pmut_fom.into_base(),
        recommended,
    ])
}
