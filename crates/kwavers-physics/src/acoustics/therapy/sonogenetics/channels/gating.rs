//! Open-probability equations for mechanosensitive channel gates.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::Array3;

use aequitas::systems::si::quantities::ThermodynamicTemperature;

use super::constants::K_B;
use super::params::{BoltzmannGatingParams, GatingModel, PressureThresholdParams};
use crate::parallel::zip_mut_with;

/// Compute per-voxel open probability using the Boltzmann two-state model.
///
/// # Formula
///
/// `P_open = 1 / (1 + exp(-A_gate * (Delta T - T_half) / (k_B * T_temp)))`
///
/// # Proof obligations
///
/// At `Delta T = T_half`, the exponent is zero and the value is `0.5`. With
/// positive absolute temperature, positive gating area, and finite tension, the
/// derivative is positive because the logistic derivative is positive and the
/// affine tension coefficient is positive.
///
/// # Errors
///
/// Returns `Err` if the absolute temperature is not strictly positive.
pub fn boltzmann_p_open(
    membrane_tension: &Array3<f64>,
    params: &BoltzmannGatingParams,
    temperature: ThermodynamicTemperature<f64>,
) -> KwaversResult<Array3<f64>> {
    if temperature.into_base() <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: "temperature".to_owned(),
            value: temperature.into_base(),
            reason: "absolute temperature must be strictly positive".to_owned(),
        }));
    }
    let kbt = K_B * temperature.into_base();
    let a = params.gating_area.into_base();
    let t_half = params.half_tension.into_base();
    let mut out = Array3::<f64>::zeros(membrane_tension.shape());
    zip_mut_with(
        out.view_mut(),
        &membrane_tension.view(),
        |p: &mut f64, &dt: &f64| {
            let exponent = -a * (dt - t_half) / kbt;
            *p = 1.0 / (1.0 + exponent.exp());
        },
    );
    Ok(out)
}

/// Compute per-voxel open probability using the sigmoidal pressure-threshold model.
///
/// # Formula
///
/// `P_open = 1 / (1 + exp(-(P_rad - P_half) / s))`
///
/// # Errors
///
/// Returns `Err` if the sigmoid steepness is not strictly positive.
pub fn pressure_threshold_p_open(
    radiation_pressure: &Array3<f64>,
    params: &PressureThresholdParams,
) -> KwaversResult<Array3<f64>> {
    if params.steepness.into_base() <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: "steepness".to_owned(),
            value: params.steepness.into_base(),
            reason: "sigmoid steepness must be strictly positive".to_owned(),
        }));
    }
    let p_half = params.half_pressure.into_base();
    let s = params.steepness.into_base();
    let mut out = Array3::<f64>::zeros(radiation_pressure.shape());
    zip_mut_with(
        out.view_mut(),
        &radiation_pressure.view(),
        |p: &mut f64, &p_rad: &f64| {
            *p = 1.0 / (1.0 + (-(p_rad - p_half) / s).exp());
        },
    );
    Ok(out)
}

/// Dispatch to the appropriate gating model.
///
/// Boltzmann models consume membrane tension and temperature; pressure-threshold
/// models consume acoustic radiation pressure.
///
/// # Errors
///
/// Propagates validation errors from the selected gating equation.
pub fn compute_p_open(
    model: &GatingModel,
    membrane_tension: &Array3<f64>,
    radiation_pressure: &Array3<f64>,
    temperature: ThermodynamicTemperature<f64>,
) -> KwaversResult<Array3<f64>> {
    match model {
        GatingModel::Boltzmann(params) => boltzmann_p_open(membrane_tension, params, temperature),
        GatingModel::PressureThreshold(params) => {
            pressure_threshold_p_open(radiation_pressure, params)
        }
    }
}
