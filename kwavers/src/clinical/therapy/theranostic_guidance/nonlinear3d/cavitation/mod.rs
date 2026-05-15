//! Rayleigh-Plesset cavitation forward map and passive inverse.
//!
//! Algorithm: each active voxel receives the local Westervelt peak pressure as
//! the acoustic forcing amplitude in the Rayleigh-Plesset ODE. The source
//! density is the maximum period-doubled radius response, which is then mapped
//! to passive receivers by a subharmonic Green operator. The inverse solves a
//! nonnegative Tikhonov problem by projected gradient descent with step bounded
//! by the Frobenius norm of the discrete operator.

mod forward;
mod helpers;
mod passive_inverse;

use ndarray::Array3;

use super::metrics::metrics_from_score;
use super::types::{Nonlinear3dAperture, Nonlinear3dConfig, Nonlinear3dVolume, VolumeReconstructionMetrics};

use forward::cavitation_source;
use helpers::{active_indices, normalize, positive_mask, unflatten};
use passive_inverse::{solve_projected_tikhonov, PassiveOperator};

#[derive(Clone, Debug)]
pub(crate) struct CavitationResult {
    pub source_density: Array3<f64>,
    pub reconstructed_density: Array3<f64>,
    pub objective_history: Vec<f64>,
    pub metrics: VolumeReconstructionMetrics,
}

pub(crate) fn run_cavitation_inverse(
    volume: &Nonlinear3dVolume,
    aperture: &Nonlinear3dAperture,
    peak_pressure: &Array3<f64>,
    config: &Nonlinear3dConfig,
) -> CavitationResult {
    let n = volume.body_mask.dim().0;
    let body = volume.body_mask.iter().copied().collect::<Vec<_>>();
    let source = cavitation_source(volume, peak_pressure, config);
    let source_vec = source.iter().copied().collect::<Vec<_>>();
    let active = active_indices(&body);
    let operator = PassiveOperator::new(volume, aperture, &active, config);
    let data = operator.apply(&source_vec);
    let inverse = solve_projected_tikhonov(&operator, &data, config);
    let mut reconstructed = vec![0.0; n * n * n];
    for (col, cell) in active.iter().enumerate() {
        reconstructed[*cell] = inverse.model[col];
    }
    let source_mask = positive_mask(&source_vec, &body);
    let score = normalize(&reconstructed, &body);
    CavitationResult {
        source_density: source,
        reconstructed_density: unflatten(&reconstructed, n),
        objective_history: inverse.objective_history,
        metrics: metrics_from_score(&score, &source_mask, &body),
    }
}
