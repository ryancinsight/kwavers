//! Construction and distance-table precomputation for [`super::VolumeOperator`].

use rayon::prelude::*;
use std::f64::consts::TAU;

use super::super::LinearBornInversionConfig;
use super::helpers::distance;
use super::{RowContext, VolumeOperator, VolumeVoxel};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_transducer::transducers::TransducerGeometry;

impl<'a> VolumeOperator<'a> {
    /// Construct the matrix-free operator over `active` voxels for the given
    /// acquisition geometry.
    ///
    /// `geometry` provides element Cartesian positions; `receiver_indices` is
    /// the geometry's per-`(source, offset)` receiver lookup
    /// (`geometry.receiver_indices(&config.receiver_offsets)`).
    /// Anatomy-specific reference quantities (`c₀`, `ρ₀`) are drawn from
    /// `config.reference_sound_speed_m_s` and `config.reference_density_kg_m3`.
    pub fn new<G: TransducerGeometry + ?Sized>(
        geometry: &G,
        receiver_indices: &[usize],
        active: &'a [VolumeVoxel],
        voxel_volume_m3: f64,
        config: &LinearBornInversionConfig,
    ) -> Self {
        let elements = geometry.elements();
        let element_count = elements.len();
        let row_contexts = build_row_contexts(receiver_indices, element_count, config);
        let n_active = active.len();

        // Pre-fill distance tables in parallel over elements.
        // Each element's n_active distances form one contiguous chunk.
        let mut elem_dist = vec![0.0f64; element_count * n_active];
        if n_active > 0 {
            elem_dist
                .par_chunks_mut(n_active)
                .enumerate()
                .for_each(|(elem_idx, chunk)| {
                    let elem = &elements[elem_idx];
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
    element_count: usize,
    config: &LinearBornInversionConfig,
) -> Vec<RowContext> {
    (0..config.measurement_count(element_count))
        .map(|row| build_row_context(row, receiver_indices, config))
        .collect()
}

fn build_row_context(
    row: usize,
    receiver_indices: &[usize],
    config: &LinearBornInversionConfig,
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
        frequency_mhz: channel_frequency_hz / MHZ_TO_HZ,
        harmonic_path_scale: second_harmonic_scale(config, frequency_hz, harmonic_order),
        attenuation_model: config.attenuation_model,
        k: TAU * channel_frequency_hz / config.reference_sound_speed_m_s,
    }
}

/// Compute the linear path-growth coefficient for the second harmonic.
///
/// From the weak shock distance `z_s = ρ₀ c₀³ / (β ω p₀)` (Blackstock 1966),
/// the second-harmonic amplitude grows as `p₂ ≈ (p₁² / 4) · path / z_s`,
/// giving a path-scale coefficient of `1 / (4 z_s)`.  Fundamental rows
/// (`harmonic_order == 1`) return 0, which disables the harmonic term in
/// [`super::kernel::VolumeOperator::row_value_for_col`].
fn second_harmonic_scale(
    config: &LinearBornInversionConfig,
    frequency_hz: f64,
    harmonic_order: usize,
) -> f64 {
    if harmonic_order == 1 {
        return 0.0;
    }
    let source_pressure_pa = config.source_pressure_mpa * MPA_TO_PA;
    let omega = TAU * frequency_hz;
    let shock_distance_m = config.reference_density_kg_m3
        * config.reference_sound_speed_m_s.powi(3)
        / (config.nonlinear_beta * omega * source_pressure_pa);
    0.25 / shock_distance_m
}
