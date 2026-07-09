//! Ultrasonic speed-of-sound shift imaging.
//!
//! This module reconstructs a 2-D map of `delta c = c - c0` from differential
//! ultrasonic travel-time shifts. For a straight ray `Gamma_r` and small sound
//! speed perturbation:
//!
//! ```text
//! delta t_r = integral_Gamma (1 / (c0 + delta c) - 1 / c0) ds
//!           = -1 / c0^2 * integral_Gamma delta c(s) ds + O(delta c^2).
//! ```
//!
//! The discrete inverse uses exact segment/pixel intersection lengths
//! `A[r, v] = length(Gamma_r intersect voxel_v)` and solves
//! `A delta_c = -c0^2 delta_t`. Dense imaging uses Tikhonov/H1 PCG; sparse
//! imaging uses the same operator with deterministic row selection and an L1
//! proximal step.

mod curved_array;
mod fixed_acquisition;
mod openpros_benchmark;
mod operator;
mod propagation;
mod ray;
mod solver;
#[cfg(test)]
mod tests;
mod types;

use leto::Array2;

use kwavers_core::error::KwaversResult;

pub use curved_array::{CurvedArray2d, CurvedArrayShiftScan};
pub use fixed_acquisition::{
    SoundSpeedShiftBatch, SoundSpeedShiftBatchConfig, SoundSpeedShiftBatchFrame,
    SoundSpeedShiftFrameSummary, SoundSpeedShiftObjectiveHistoryPolicy, SoundSpeedShiftPlan,
};
pub use openpros_benchmark::{
    openpros_shift_benchmark_case, run_openpros_shift_benchmark, OpenProsShiftBenchmarkCase,
    OpenProsShiftBenchmarkConfig, OpenProsShiftBenchmarkResult, OpenProsShiftReconstructionMetrics,
    OpenProsWaveformExpectation, OPENPROS_PAPER_ID,
};
use operator::SoundSpeedShiftOperator;
use solver::{solve_shift, solve_shift_with_metrics, SoundSpeedShiftSolverMetrics};
pub use types::{
    ShiftPrior, ShiftPropagation, ShiftSampling, ShiftSensitivity, SoundSpeedShiftConfig,
    SoundSpeedShiftImage, SoundSpeedShiftSample, SoundSpeedShiftWorkspace,
    CURVED_RAY_SOUND_SPEED_SHIFT_MODEL, FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL,
    SOUND_SPEED_SHIFT_MODEL,
};

/// Reconstruct a speed-of-sound shift image from measured travel-time shifts.
///
/// # Errors
/// Returns [`kwavers_core::error::KwaversError`] when geometry, sampling,
/// regularization, or active-mask inputs violate the reconstruction contract.
pub fn reconstruct_sound_speed_shift(
    samples: &[SoundSpeedShiftSample],
    active_mask: &Array2<bool>,
    config: SoundSpeedShiftConfig,
) -> KwaversResult<SoundSpeedShiftImage> {
    let mut workspace = SoundSpeedShiftWorkspace::new();
    reconstruct_sound_speed_shift_with_workspace(samples, active_mask, config, &mut workspace)
}

/// Reconstruct a speed-of-sound shift image using caller-owned scratch buffers.
///
/// Reusing `workspace` across repeated reconstructions preserves PCG/ISTA
/// buffer capacities for compatible or smaller domains.
///
/// # Errors
/// Returns [`kwavers_core::error::KwaversError`] when geometry, sampling,
/// regularization, or active-mask inputs violate the reconstruction contract.
pub fn reconstruct_sound_speed_shift_with_workspace(
    samples: &[SoundSpeedShiftSample],
    active_mask: &Array2<bool>,
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) -> KwaversResult<SoundSpeedShiftImage> {
    let operator = SoundSpeedShiftOperator::new(samples, active_mask, config)?;
    let data = operator.rhs_from_sample_time_shifts(samples, config.reference_sound_speed_m_s);
    Ok(reconstruct_from_operator(
        &operator,
        &data,
        samples.len(),
        config,
        workspace,
    ))
}

/// Predict travel-time shifts for a known speed-of-sound shift image.
///
/// This is the forward contract used for differential validation:
/// `delta t = -A delta_c / c0^2`.
///
/// # Errors
/// Returns [`kwavers_core::error::KwaversError`] when input geometry or the
/// supplied image/mask shapes are invalid.
pub fn predict_sound_speed_time_shifts(
    sound_speed_shift_m_s: &Array2<f64>,
    samples: &[SoundSpeedShiftSample],
    active_mask: &Array2<bool>,
    config: SoundSpeedShiftConfig,
) -> KwaversResult<Vec<f64>> {
    if sound_speed_shift_m_s.dim() != active_mask.dim() {
        return Err(kwavers_core::error::KwaversError::DimensionMismatch(
            format!(
                "speed-shift image shape {:?} does not match active-mask shape {:?}",
                sound_speed_shift_m_s.dim(),
                active_mask.dim()
            ),
        ));
    }

    let dense_config = SoundSpeedShiftConfig {
        sampling: ShiftSampling::Dense,
        iterations: config.iterations.max(1),
        ..config
    };
    let operator = SoundSpeedShiftOperator::new(samples, active_mask, dense_config)?;
    let model = operator.model_from_image(sound_speed_shift_m_s);
    let mut path_integrals = vec![0.0; operator.rows()];
    operator.matvec(&model, &mut path_integrals);
    let c0_sq = config.reference_sound_speed_m_s * config.reference_sound_speed_m_s;

    Ok(path_integrals
        .into_iter()
        .map(|value| -value / c0_sq)
        .collect())
}

fn reconstruct_from_operator(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    rows_available: usize,
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) -> SoundSpeedShiftImage {
    solve_operator_frame(operator, data, config, workspace);

    SoundSpeedShiftImage {
        sound_speed_shift_m_s: solved_image_from_operator(operator, workspace),
        objective_history: workspace.objective_history.clone(),
        rows_used: operator.rows(),
        rows_available,
        active_voxels: operator.cols(),
        model_family: config.model_family(),
        sampling: config.sampling,
        prior: config.prior,
    }
}

fn solve_operator_frame(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
) {
    solve_shift(operator, data, config, workspace);
}

pub(in crate::reconstruction::sound_speed_shift) fn solve_operator_frame_with_metrics(
    operator: &SoundSpeedShiftOperator,
    data: &[f64],
    config: SoundSpeedShiftConfig,
    workspace: &mut SoundSpeedShiftWorkspace,
    metrics: &SoundSpeedShiftSolverMetrics,
) {
    solve_shift_with_metrics(operator, data, config, workspace, metrics);
}

fn solved_image_from_operator(
    operator: &SoundSpeedShiftOperator,
    workspace: &SoundSpeedShiftWorkspace,
) -> Array2<f64> {
    operator.image_from_model(&workspace.solution)
}

pub(in crate::reconstruction::sound_speed_shift) fn solved_image_from_operator_into(
    operator: &SoundSpeedShiftOperator,
    workspace: &SoundSpeedShiftWorkspace,
    image: &mut Array2<f64>,
) {
    operator.image_from_model_into(&workspace.solution, image);
}
