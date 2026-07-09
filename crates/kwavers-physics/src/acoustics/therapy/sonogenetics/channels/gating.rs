//! Open-probability equations for mechanosensitive channel gates.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::Array3;

use super::constants::K_B;
use super::params::{BoltzmannGatingParams, GatingModel, PressureThresholdParams};
use crate::parallel::zip_mut_ref;

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
/// Returns `Err` if `temperature_k <= 0`.
pub fn boltzmann_p_open(
    membrane_tension: &Array3<f64>,
    params: &BoltzmannGatingParams,
    temperature_k: f64,
) -> KwaversResult<Array3<f64>> {
    if temperature_k <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: "temperature_k".to_owned(),
            value: temperature_k,
            reason: "absolute temperature must be strictly positive".to_owned(),
        }));
    }
    let kbt = K_B * temperature_k;
    let a = params.gating_area_m2;
    let t_half = params.half_tension_n_per_m;
    let mut out = Array3::<f64>::zeros(membrane_tension.shape());
    zip_mut_ref(
        out.view_mut(),
        membrane_tension.view(),
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
/// Returns `Err` if `steepness_pa <= 0`.
pub fn pressure_threshold_p_open(
    radiation_pressure: &Array3<f64>,
    params: &PressureThresholdParams,
) -> KwaversResult<Array3<f64>> {
    if params.steepness_pa <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: "steepness_pa".to_owned(),
            value: params.steepness_pa,
            reason: "sigmoid steepness must be strictly positive".to_owned(),
        }));
    }
    let p_half = params.half_pressure_pa;
    let s = params.steepness_pa;
    let mut out = Array3::<f64>::zeros(radiation_pressure.shape());
    zip_mut_ref(
        out.view_mut(),
        radiation_pressure.view(),
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
    temperature_k: f64,
) -> KwaversResult<Array3<f64>> {
    match model {
        GatingModel::Boltzmann(params) => boltzmann_p_open(membrane_tension, params, temperature_k),
        GatingModel::PressureThreshold(params) => {
            pressure_threshold_p_open(radiation_pressure, params)
        }
    }
}
