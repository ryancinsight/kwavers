//! Matrix-free 3-D Born inversion for CT-derived brain speed contrast.
//!
//! Clinical adapter: composes the brain anatomy/transducer-geometry with the
//! generic solver-layer kernels in
//! [`kwavers_solver::inverse::linear_born_inversion`].

use kwavers_core::error::KwaversResult;
use kwavers_math::statistics::{normalized_rmse, pearson, percentile_range};
use kwavers_transducer::transducers::TransducerGeometry;
use kwavers_solver::inverse::linear_born_inversion::{
    high_pass_enhance_volume, pcg_invert, VolumeOperator, VolumeVoxel,
};
use ndarray::Array3;

use super::{
    config::{TranscranialUstBornInversionConfig, SOUND_SPEED_TISSUE},
    transducer::TranscranialBowlGeometry,
    volume::AcousticVolume,
};

/// Quality metrics for the reconstructed 3-D brain volume.
#[derive(Clone, Debug)]
pub struct TranscranialUstBornInversionVolumeMetrics {
    pub initial_objective: f64,
    pub final_objective: f64,
    pub objective_reduction_fraction: f64,
    pub migration_pearson_correlation: f64,
    pub migration_normalized_rmse: f64,
    pub migration_dynamic_range_m_s: f64,
    pub pearson_correlation: f64,
    pub normalized_rmse: f64,
    pub enhanced_dynamic_range_m_s: f64,
    pub active_voxels: usize,
    pub measurements: usize,
    pub continuation_stages: usize,
    pub target_dynamic_range_m_s: f64,
    pub reconstruction_dynamic_range_m_s: f64,
}

/// Result arrays and diagnostics from a 3-D encoded transcranial bowl inversion.
#[derive(Clone, Debug)]
pub struct TranscranialUstBornInversionVolumeResult {
    pub ct_hu: Array3<f64>,
    pub target_sound_speed_m_s: Array3<f64>,
    pub initial_sound_speed_m_s: Array3<f64>,
    pub migration_sound_speed_m_s: Array3<f64>,
    pub reconstruction_sound_speed_m_s: Array3<f64>,
    pub enhanced_reconstruction_sound_speed_m_s: Array3<f64>,
    pub brain_mask: Array3<bool>,
    pub skull_mask: Array3<bool>,
    pub synthetic_data: Vec<f64>,
    pub residual_history: Vec<f64>,
    pub metrics: TranscranialUstBornInversionVolumeMetrics,
}

/// Reconstruct brain sound-speed contrast as one coupled 3-D volume.
pub fn reconstruct_brain_volume(
    medium: &AcousticVolume,
    config: &TranscranialUstBornInversionConfig,
) -> KwaversResult<TranscranialUstBornInversionVolumeResult> {
    config.validate()?;
    let linear = &config.linear;
    let geometry = TranscranialBowlGeometry::from_aperture(
        config.element_count,
        config.radius_m,
        config.aperture,
    )?;
    let receiver_indices = geometry.receiver_indices(&linear.receiver_offsets);
    let active = active_voxels(medium);
    let operator = VolumeOperator::new(
        &geometry,
        &receiver_indices,
        &active,
        medium.spacing_m * medium.spacing_m * medium.spacing_m,
        linear,
    );
    let nrows = config.measurement_count();
    let all_rows: Vec<usize> = (0..nrows).collect();
    let row_norms = operator.row_norms();
    let data = operator.data_from_target(&row_norms);
    let migration_model = operator.migration(&data, &all_rows, &row_norms, linear);
    let inversion = pcg_invert(
        &operator,
        &data,
        &row_norms,
        linear,
        &active,
        medium.sound_speed_m_s.dim(),
    );

    let migration = fill_sound_speed_volume(medium, &active, &migration_model);
    let reconstruction = fill_sound_speed_volume(medium, &active, &inversion.model);
    let enhanced = high_pass_enhance_volume(
        &reconstruction,
        &medium.brain_mask,
        linear.enhancement_gain,
        SOUND_SPEED_TISSUE,
    );

    let target: Vec<f64> = active
        .iter()
        .map(|v| SOUND_SPEED_TISSUE * (1.0 + v.target_contrast))
        .collect();
    let recon: Vec<f64> = active
        .iter()
        .enumerate()
        .map(|(idx, _)| SOUND_SPEED_TISSUE * (1.0 + inversion.model[idx]))
        .collect();
    let migration_values: Vec<f64> = active
        .iter()
        .enumerate()
        .map(|(idx, _)| SOUND_SPEED_TISSUE * (1.0 + migration_model[idx]))
        .collect();
    let enhanced_values: Vec<f64> = active
        .iter()
        .map(|voxel| enhanced[[voxel.ix, voxel.iy, voxel.iz]])
        .collect();
    let initial_objective = inversion.history.first().copied().unwrap_or(0.0);
    let final_objective = inversion
        .history
        .last()
        .copied()
        .unwrap_or(initial_objective);
    let objective_reduction_fraction = if initial_objective > 0.0 {
        ((initial_objective - final_objective) / initial_objective).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(TranscranialUstBornInversionVolumeResult {
        ct_hu: medium.ct_hu.clone(),
        target_sound_speed_m_s: medium.sound_speed_m_s.clone(),
        initial_sound_speed_m_s: medium.initial_sound_speed_m_s.clone(),
        migration_sound_speed_m_s: migration,
        reconstruction_sound_speed_m_s: reconstruction,
        enhanced_reconstruction_sound_speed_m_s: enhanced,
        brain_mask: medium.brain_mask.clone(),
        skull_mask: medium.skull_mask.clone(),
        synthetic_data: data,
        residual_history: inversion.history,
        metrics: TranscranialUstBornInversionVolumeMetrics {
            initial_objective,
            final_objective,
            objective_reduction_fraction,
            migration_pearson_correlation: pearson(&target, &migration_values),
            migration_normalized_rmse: normalized_rmse(&target, &migration_values),
            migration_dynamic_range_m_s: percentile_range(migration_values),
            pearson_correlation: pearson(&target, &recon),
            normalized_rmse: normalized_rmse(&target, &recon),
            enhanced_dynamic_range_m_s: percentile_range(enhanced_values),
            active_voxels: active.len(),
            measurements: nrows,
            continuation_stages: inversion.stages,
            target_dynamic_range_m_s: percentile_range(target),
            reconstruction_dynamic_range_m_s: percentile_range(recon),
        },
    })
}

fn active_voxels(medium: &AcousticVolume) -> Vec<VolumeVoxel> {
    let (nx, ny, nz) = medium.sound_speed_m_s.dim();
    let cx = (nx - 1) as f64 / 2.0;
    let cy = (ny - 1) as f64 / 2.0;
    let cz = (nz - 1) as f64 / 2.0;
    let mut active = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                if medium.brain_mask[[ix, iy, iz]] {
                    let speed = medium.sound_speed_m_s[[ix, iy, iz]];
                    active.push(VolumeVoxel {
                        ix,
                        iy,
                        iz,
                        x_m: (ix as f64 - cx) * medium.spacing_m,
                        y_m: (iy as f64 - cy) * medium.spacing_m,
                        z_m: (iz as f64 - cz) * medium.spacing_m,
                        target_contrast: speed / SOUND_SPEED_TISSUE - 1.0,
                        attenuation_np_per_m_mhz: medium.attenuation_np_per_m_mhz[[ix, iy, iz]],
                    });
                }
            }
        }
    }
    active
}

fn fill_sound_speed_volume(
    medium: &AcousticVolume,
    active: &[VolumeVoxel],
    contrast: &[f64],
) -> Array3<f64> {
    let mut out = medium.initial_sound_speed_m_s.clone();
    for (voxel, value) in active.iter().zip(contrast) {
        out[[voxel.ix, voxel.iy, voxel.iz]] = SOUND_SPEED_TISSUE * (1.0 + value);
    }
    out
}
