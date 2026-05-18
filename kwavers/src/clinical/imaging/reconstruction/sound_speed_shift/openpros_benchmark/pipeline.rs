//! OpenPros-style benchmark construction and execution.

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use crate::core::error::{KwaversError, KwaversResult};

use super::super::{
    predict_sound_speed_time_shifts, ShiftPrior, ShiftPropagation, ShiftSampling, ShiftSensitivity,
    SoundSpeedShiftConfig, SoundSpeedShiftPlan, SoundSpeedShiftWorkspace,
};
use super::acquisition;
use super::metrics::metrics_for;
use super::phantom;
use super::types::{
    OpenProsShiftBenchmarkCase, OpenProsShiftBenchmarkConfig, OpenProsShiftBenchmarkResult,
};

/// Build the deterministic OpenPros-style limited-view benchmark case.
///
/// # Errors
/// Returns [`KwaversError`] when the reduced fixture configuration would create
/// an invalid grid, acquisition, sparse sampling policy, or waveform contract.
pub fn openpros_shift_benchmark_case(
    config: &OpenProsShiftBenchmarkConfig,
) -> KwaversResult<OpenProsShiftBenchmarkCase> {
    validate_config(config)?;
    let active_mask = phantom::active_mask(config);
    let truth_shift_m_s = phantom::shift_phantom(config);
    let samples = acquisition::samples(config);
    let dense_config = reconstruction_config(config, ShiftSampling::Dense, ShiftPrior::Dense);
    let frame_time_shifts_s =
        predict_sound_speed_time_shifts(&truth_shift_m_s, &samples, &active_mask, dense_config)?;
    let sparse_config = reconstruction_config(
        config,
        ShiftSampling::Sparse {
            stride: config.sparse_stride,
            offset: 0,
        },
        ShiftPrior::Sparse,
    );

    Ok(OpenProsShiftBenchmarkCase {
        active_mask,
        truth_shift_m_s,
        samples,
        frame_time_shifts_s,
        dense_config,
        sparse_config,
        waveform: config.waveform_expectation(),
    })
}

/// Run one dense and one sparse reconstruction on the same benchmark frame.
///
/// Dense and sparse branches differ only by `ShiftSampling` and `ShiftPrior`;
/// both branches use the existing fixed-acquisition reconstruction plan.
///
/// # Errors
/// Returns [`KwaversError`] when benchmark construction or either
/// reconstruction branch violates the underlying speed-shift contracts.
pub fn run_openpros_shift_benchmark(
    config: &OpenProsShiftBenchmarkConfig,
) -> KwaversResult<OpenProsShiftBenchmarkResult> {
    let case = openpros_shift_benchmark_case(config)?;
    let dense_plan =
        SoundSpeedShiftPlan::new(case.samples.clone(), &case.active_mask, case.dense_config)?;
    let sparse_plan =
        SoundSpeedShiftPlan::new(case.samples.clone(), &case.active_mask, case.sparse_config)?;
    let mut dense_workspace = SoundSpeedShiftWorkspace::new();
    let mut sparse_workspace = SoundSpeedShiftWorkspace::new();
    let dense_reconstruction =
        dense_plan.reconstruct_with_workspace(&case.frame_time_shifts_s, &mut dense_workspace)?;
    let sparse_reconstruction =
        sparse_plan.reconstruct_with_workspace(&case.frame_time_shifts_s, &mut sparse_workspace)?;
    let dense_metrics = metrics_for(
        &dense_reconstruction,
        &case.truth_shift_m_s,
        &case.active_mask,
        &dense_plan,
    );
    let sparse_metrics = metrics_for(
        &sparse_reconstruction,
        &case.truth_shift_m_s,
        &case.active_mask,
        &sparse_plan,
    );

    Ok(OpenProsShiftBenchmarkResult {
        dense_reconstruction,
        sparse_reconstruction,
        dense_metrics,
        sparse_metrics,
        waveform: case.waveform,
    })
}

fn reconstruction_config(
    config: &OpenProsShiftBenchmarkConfig,
    sampling: ShiftSampling,
    prior: ShiftPrior,
) -> SoundSpeedShiftConfig {
    let reference_sound_speed_m_s = SOUND_SPEED_TISSUE;
    let wavelength_m = reference_sound_speed_m_s / config.waveform_expectation().peak_frequency_hz;
    SoundSpeedShiftConfig {
        reference_sound_speed_m_s,
        spacing_m: config.spacing_m(),
        iterations: match prior {
            ShiftPrior::Dense | ShiftPrior::Lsqr { .. } => config.dense_iterations,
            ShiftPrior::Sparse => config.sparse_iterations,
        },
        tikhonov_weight: config.tikhonov_weight,
        smoothness_weight: config.smoothness_weight,
        sparsity_weight: match prior {
            ShiftPrior::Sparse => config.sparsity_weight,
            ShiftPrior::Dense | ShiftPrior::Lsqr { .. } => 0.0,
        },
        sampling,
        prior,
        propagation: ShiftPropagation::StraightRay,
        sensitivity: ShiftSensitivity::FiniteFrequency {
            wavelength_m,
            support_radius_m: config.spacing_m(),
        },
    }
}

fn validate_config(config: &OpenProsShiftBenchmarkConfig) -> KwaversResult<()> {
    if config.spatial_decimation == 0 {
        return Err(KwaversError::InvalidInput(
            "OpenPros benchmark spatial_decimation must be positive".to_owned(),
        ));
    }
    if config.source_count_per_probe == 0 || config.receiver_count_per_probe < 2 {
        return Err(KwaversError::InvalidInput(
            "OpenPros benchmark requires at least one source and two receivers per probe"
                .to_owned(),
        ));
    }
    if config.sparse_stride < 2 {
        return Err(KwaversError::InvalidInput(
            "OpenPros benchmark sparse_stride must be at least 2".to_owned(),
        ));
    }
    let weights = [
        ("tikhonov_weight", config.tikhonov_weight),
        ("smoothness_weight", config.smoothness_weight),
        ("sparsity_weight", config.sparsity_weight),
    ];
    for (name, value) in weights {
        if !value.is_finite() || value < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "OpenPros benchmark {name} must be finite and nonnegative, got {value}"
            )));
        }
    }
    Ok(())
}
