//! Construction and distance-table precomputation for [`super::VolumeOperator`].

use rayon::prelude::*;
use std::f64::consts::TAU;

use super::super::config::{TranscranialUstBornInversionConfig, C_BRAIN_REF_M_S};
use super::super::transducer::TranscranialBowlGeometry;
use super::helpers::distance;
use super::{RowContext, VolumeOperator, VolumeVoxel, C_TISSUE_DENSITY_KG_M3};

impl<'a> VolumeOperator<'a> {
    pub fn new(
        geometry: TranscranialBowlGeometry,
        receiver_indices: Vec<usize>,
        active: &'a [VolumeVoxel],
        voxel_volume_m3: f64,
        config: &TranscranialUstBornInversionConfig,
    ) -> Self {
        let row_contexts = build_row_contexts(&receiver_indices, config);
        let n_active = active.len();
        let n_elements = geometry.elements.len();

        // Pre-fill distance tables in parallel over elements.
        // Each element's n_active distances form one contiguous chunk.
        let mut elem_dist = vec![0.0f64; n_elements * n_active];
        if n_active > 0 {
            elem_dist
                .par_chunks_mut(n_active)
                .enumerate()
                .for_each(|(elem_idx, chunk)| {
                    let elem = &geometry.elements[elem_idx];
                    for (col, voxel) in active.iter().enumerate() {
                        chunk[col] = distance(
                            elem.x_m, elem.y_m, elem.z_m, voxel.x_m, voxel.y_m, voxel.z_m,
                        );
                    }
                });
        }
        // Parallelise the sqrt table: n_elements × n_active entries.
        // Serial `iter().map(sqrt)` is a bottleneck at clinical grid sizes
        // (1024-element × 50 K active voxels ≈ 51 M entries).
        let elem_sqrt_dist: Vec<f64> = elem_dist.par_iter().map(|d| d.sqrt()).collect();

        Self {
            active,
            voxel_volume_m3,
            row_contexts,
            elem_dist,
            elem_sqrt_dist,
            n_active,
        }
    }
}

fn build_row_contexts(
    receiver_indices: &[usize],
    config: &TranscranialUstBornInversionConfig,
) -> Vec<RowContext> {
    (0..config.measurement_count())
        .map(|row| build_row_context(row, receiver_indices, config))
        .collect()
}

fn build_row_context(
    row: usize,
    receiver_indices: &[usize],
    config: &TranscranialUstBornInversionConfig,
) -> RowContext {
    let harmonic_count = config.harmonic_count();
    let offset_count = config.receiver_offsets.len();
    let frequency_count = config.frequencies_hz.len();
    let source_idx = row / (offset_count * frequency_count * harmonic_count);
    let offset_idx = (row / (frequency_count * harmonic_count)) % offset_count;
    let frequency_idx = (row / harmonic_count) % frequency_count;
    let harmonic_idx = row % harmonic_count;
    let frequency_hz = config.frequencies_hz[frequency_idx];
    let harmonic_order = harmonic_idx + 1;
    let channel_frequency_hz = harmonic_order as f64 * frequency_hz;
    let receiver_idx = receiver_indices[source_idx * offset_count + offset_idx];
    RowContext {
        source_idx,
        receiver_idx,
        frequency_mhz: channel_frequency_hz * 1.0e-6,
        harmonic_path_scale: second_harmonic_scale(config, frequency_hz, harmonic_order),
        attenuation_model: config.attenuation_model,
        k: TAU * channel_frequency_hz / C_BRAIN_REF_M_S,
    }
}

/// Compute the linear path-growth coefficient for the second harmonic.
///
/// From the weak shock distance `z_s = ρ c³ / (β ω p₀)` (Blackstock 1966),
/// the second-harmonic amplitude grows as `p₂ ≈ (p₁² / 4) · path / z_s`,
/// giving a path-scale coefficient of `1 / (4 z_s)`.  Fundamental rows
/// (`harmonic_order == 1`) return 0, which disables the harmonic term in
/// [`super::kernel::VolumeOperator::row_value_for_col`].
fn second_harmonic_scale(
    config: &TranscranialUstBornInversionConfig,
    frequency_hz: f64,
    harmonic_order: usize,
) -> f64 {
    if harmonic_order == 1 {
        return 0.0;
    }
    let source_pressure_pa = config.source_pressure_mpa * 1.0e6;
    let omega = TAU * frequency_hz;
    let shock_distance_m = C_TISSUE_DENSITY_KG_M3 * C_BRAIN_REF_M_S.powi(3)
        / (config.nonlinear_beta * omega * source_pressure_pa);
    0.25 / shock_distance_m
}
