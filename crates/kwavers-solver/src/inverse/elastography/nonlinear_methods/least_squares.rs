//! Iterative Gauss-Newton nonlinear least squares inversion.
//!
//! Fits measured (A₁, A₂) to the shear-wave harmonic model
//! (Nocedal & Wright 2006, §10.3; Rénier 2008).

use super::super::config::NonlinearInversionConfig;
use super::helpers::{
    a_landau, ba_from_beta_s, beta_s_from_amplitudes, forward_model, forward_model_derivative,
    shear_modulus,
};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::NonlinearParameterMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;
use leto::Array3;

/// Iterative nonlinear least squares inversion (Gauss-Newton).
///
/// # Algorithm
///
/// 1. Seed from harmonic ratio (warm start).
/// 2. Gauss-Newton: Δ(B/A) = Jᵀr / JᵀJ until |Δ(B/A)| < tolerance.
///
/// # References
///
/// - Nocedal & Wright (2006). *Numerical Optimization*, §10.3.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn nonlinear_least_squares_inversion(
    harmonic_field: &HarmonicDisplacementField,
    _grid: &Grid,
    config: &NonlinearInversionConfig,
) -> KwaversResult<NonlinearParameterMap> {
    let [nx, ny, nz] = harmonic_field.fundamental_magnitude.shape();

    let mut nonlinearity_parameter = Array3::zeros([nx, ny, nz]);
    let mut nonlinearity_uncertainty = Array3::zeros([nx, ny, nz]);
    let mut estimation_quality = Array3::zeros([nx, ny, nz]);

    let mut elastic_constants = vec![
        Array3::zeros([nx, ny, nz]),
        Array3::zeros([nx, ny, nz]),
        Array3::zeros([nx, ny, nz]),
        Array3::zeros([nx, ny, nz]),
    ];

    let mu = shear_modulus(config);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let measured_a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                let measured_a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];

                if measured_a1 < 1e-12 {
                    nonlinearity_parameter[[i, j, k]] = 0.0;
                    nonlinearity_uncertainty[[i, j, k]] = 1.0;
                    estimation_quality[[i, j, k]] = 0.0;
                    continue;
                }

                let mut ba_estimate = beta_s_from_amplitudes(measured_a1, measured_a2, config)
                    .map_or(5.0, ba_from_beta_s)
                    .clamp(0.0, 20.0);

                let mut converged = false;
                for _iteration in 0..config.max_iterations {
                    let (predicted_a1, predicted_a2) =
                        forward_model(ba_estimate, measured_a1, config);

                    let residual_a1 = measured_a1 - predicted_a1;
                    let residual_a2 = measured_a2 - predicted_a2;

                    let (da1_dba, da2_dba) =
                        forward_model_derivative(ba_estimate, measured_a1, config);

                    let jt_j = da1_dba.mul_add(da1_dba, da2_dba * da2_dba);
                    if jt_j.abs() < f64::EPSILON {
                        break;
                    }
                    let delta_ba = residual_a1.mul_add(da1_dba, residual_a2 * da2_dba) / jt_j;
                    ba_estimate = (ba_estimate + delta_ba).clamp(0.0, 20.0);

                    if delta_ba.abs() < config.tolerance {
                        converged = true;
                        break;
                    }
                }

                nonlinearity_parameter[[i, j, k]] = ba_estimate;
                nonlinearity_uncertainty[[i, j, k]] = if converged { 0.1 } else { 1.0 };
                estimation_quality[[i, j, k]] = if converged { 0.9 } else { 0.5 };

                let beta = ba_estimate / 2.0 + 1.0;
                let al = a_landau(mu, beta);
                elastic_constants[0][[i, j, k]] = al;
                elastic_constants[1][[i, j, k]] = al / 3.0;
                elastic_constants[2][[i, j, k]] = al / 9.0;
                elastic_constants[3][[i, j, k]] = al / 27.0;
            }
        }
    }

    Ok(NonlinearParameterMap {
        nonlinearity_parameter,
        elastic_constants,
        nonlinearity_uncertainty,
        estimation_quality,
    })
}
