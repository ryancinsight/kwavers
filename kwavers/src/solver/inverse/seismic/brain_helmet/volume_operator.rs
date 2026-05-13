//! Matrix-free finite-frequency operator for 3-D helmet inversion.

use rayon::prelude::*;
use std::f64::consts::TAU;

use super::{
    config::{BrainHelmetFwiConfig, C_BRAIN_REF_M_S},
    transducer::{ElementPosition, HelmetHemisphereGeometry},
};

const C_TISSUE_DENSITY_KG_M3: f64 = 1000.0;

#[derive(Clone, Copy, Debug)]
pub(super) struct VolumeVoxel {
    pub ix: usize,
    pub iy: usize,
    pub iz: usize,
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
    pub target_contrast: f64,
    pub attenuation_np_per_m_mhz: f64,
}

pub(super) struct VolumeOperator<'a> {
    active: &'a [VolumeVoxel],
    voxel_volume_m3: f64,
    row_contexts: Vec<RowContext>,
}

#[derive(Clone, Copy)]
struct RowContext {
    source: ElementPosition,
    receiver: ElementPosition,
    frequency_mhz: f64,
    harmonic_path_scale: f64,
    attenuation_model: bool,
    k: f64,
}

impl<'a> VolumeOperator<'a> {
    pub fn new(
        geometry: HelmetHemisphereGeometry,
        receiver_indices: Vec<usize>,
        active: &'a [VolumeVoxel],
        voxel_volume_m3: f64,
        config: &BrainHelmetFwiConfig,
    ) -> Self {
        let row_contexts = build_row_contexts(&geometry, &receiver_indices, config);
        Self {
            active,
            voxel_volume_m3,
            row_contexts,
        }
    }

    pub fn row_norms(&self) -> Vec<f64> {
        (0..self.row_contexts.len())
            .into_par_iter()
            .map(|row| self.row_norm(row))
            .collect()
    }

    pub fn data_from_target(&self, row_norms: &[f64]) -> Vec<f64> {
        (0..self.row_contexts.len())
            .into_par_iter()
            .map(|row| {
                self.project_row_with_norm(row, row_norms[row], |col| {
                    self.active[col].target_contrast
                })
            })
            .collect()
    }

    pub fn diagonal(
        &self,
        rows: &[usize],
        row_norms: &[f64],
        config: &BrainHelmetFwiConfig,
    ) -> Vec<f64> {
        let ncols = self.active.len();
        let mut diagonal = vec![config.regularization.max(1.0e-12); ncols];
        let partials: Vec<Vec<f64>> = rows
            .par_chunks(row_chunk_len(rows.len()))
            .map(|chunk| {
                let mut partial = vec![0.0; ncols];
                for &row in chunk {
                    let norm = row_norms[row];
                    if norm == 0.0 {
                        continue;
                    }
                    let row_context = self.row_context(row);
                    for (col, voxel) in self.active.iter().enumerate() {
                        let value = self.row_value_from_context(&row_context, voxel) / norm;
                        partial[col] += value * value;
                    }
                }
                partial
            })
            .collect();
        add_partials(&mut diagonal, partials);
        diagonal
    }

    pub fn migration(
        &self,
        data: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        config: &BrainHelmetFwiConfig,
    ) -> Vec<f64> {
        let ncols = self.active.len();
        let diagonal = self.diagonal(rows, row_norms, config);
        let mut adjoint = vec![0.0; ncols];
        let partials: Vec<Vec<f64>> = rows
            .par_chunks(row_chunk_len(rows.len()))
            .map(|chunk| {
                let mut partial = vec![0.0; ncols];
                for &row in chunk {
                    let norm = row_norms[row];
                    if norm == 0.0 {
                        continue;
                    }
                    let row_context = self.row_context(row);
                    for (col, voxel) in self.active.iter().enumerate() {
                        partial[col] +=
                            self.row_value_from_context(&row_context, voxel) * data[row] / norm;
                    }
                }
                partial
            })
            .collect();
        add_partials(&mut adjoint, partials);
        adjoint
            .into_iter()
            .zip(diagonal)
            .map(|(value, diag)| (value / diag).clamp(config.contrast_min, config.contrast_max))
            .collect()
    }

    pub fn objective(
        &self,
        data: &[f64],
        model: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        regularization: f64,
    ) -> f64 {
        let data_misfit = rows
            .par_chunks(row_chunk_len(rows.len()))
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&row| {
                        let prediction =
                            self.project_row_with_norm(row, row_norms[row], |col| model[col]);
                        let residual = data[row] - prediction;
                        0.5 * residual * residual
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        data_misfit + 0.5 * regularization * model.iter().map(|v| v * v).sum::<f64>()
    }

    pub fn normal_residual(
        &self,
        data: &[f64],
        model: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        regularization: f64,
    ) -> Vec<f64> {
        let ncols = self.active.len();
        let mut residual = vec![0.0; ncols];
        let partials: Vec<Vec<f64>> = rows
            .par_chunks(row_chunk_len(rows.len()))
            .map(|chunk| {
                let mut partial = vec![0.0; ncols];
                let mut row_values = vec![0.0; ncols];
                for &row in chunk {
                    let norm = row_norms[row];
                    if norm == 0.0 {
                        continue;
                    }
                    self.fill_row_values(row, &mut row_values);
                    let prediction = row_values
                        .iter()
                        .zip(model)
                        .map(|(row_value, model_value)| row_value * model_value / norm)
                        .sum::<f64>();
                    let row_residual = data[row] - prediction;
                    for (partial_value, row_value) in partial.iter_mut().zip(&row_values) {
                        *partial_value += row_value * row_residual / norm;
                    }
                }
                partial
            })
            .collect();
        add_partials(&mut residual, partials);
        for (value, model_value) in residual.iter_mut().zip(model) {
            *value -= regularization * model_value;
        }
        residual
    }

    pub fn apply_normal(
        &self,
        vector: &[f64],
        rows: &[usize],
        row_norms: &[f64],
        regularization: f64,
    ) -> Vec<f64> {
        let ncols = self.active.len();
        let mut out = vec![0.0; ncols];
        let partials: Vec<Vec<f64>> = rows
            .par_chunks(row_chunk_len(rows.len()))
            .map(|chunk| {
                let mut partial = vec![0.0; ncols];
                let mut row_values = vec![0.0; ncols];
                for &row in chunk {
                    let norm = row_norms[row];
                    if norm == 0.0 {
                        continue;
                    }
                    self.fill_row_values(row, &mut row_values);
                    let projection = row_values
                        .iter()
                        .zip(vector)
                        .map(|(row_value, vector_value)| row_value * vector_value / norm)
                        .sum::<f64>();
                    for (partial_value, row_value) in partial.iter_mut().zip(&row_values) {
                        *partial_value += row_value * projection / norm;
                    }
                }
                partial
            })
            .collect();
        add_partials(&mut out, partials);
        for (value, vector_value) in out.iter_mut().zip(vector) {
            *value += regularization * vector_value;
        }
        out
    }

    fn project_row_with_norm<F>(&self, row: usize, norm: f64, value_at: F) -> f64
    where
        F: Fn(usize) -> f64 + Sync,
    {
        let row_context = self.row_context(row);
        self.active
            .iter()
            .enumerate()
            .map(|(col, voxel)| {
                self.row_value_from_context(&row_context, voxel) * value_at(col) / norm
            })
            .sum()
    }

    fn row_norm(&self, row: usize) -> f64 {
        let row_context = self.row_context(row);
        self.active
            .iter()
            .map(|voxel| {
                let value = self.row_value_from_context(&row_context, voxel);
                value * value
            })
            .sum::<f64>()
            .sqrt()
    }

    fn fill_row_values(&self, row: usize, out: &mut [f64]) {
        let row_context = self.row_context(row);
        for (value, voxel) in out.iter_mut().zip(self.active) {
            *value = self.row_value_from_context(&row_context, voxel);
        }
    }

    fn row_context(&self, row: usize) -> RowContext {
        self.row_contexts[row]
    }

    fn row_value_from_context(&self, row_context: &RowContext, voxel: &VolumeVoxel) -> f64 {
        let ds = distance(
            row_context.source.x_m,
            row_context.source.y_m,
            row_context.source.z_m,
            voxel.x_m,
            voxel.y_m,
            voxel.z_m,
        );
        let dr = distance(
            row_context.receiver.x_m,
            row_context.receiver.y_m,
            row_context.receiver.z_m,
            voxel.x_m,
            voxel.y_m,
            voxel.z_m,
        );
        let path_m = ds + dr;
        let spreading = (ds * dr).sqrt().max(1.0e-6);
        let attenuation = if row_context.attenuation_model {
            (-(voxel.attenuation_np_per_m_mhz * path_m) * row_context.frequency_mhz).exp()
        } else {
            1.0
        };
        let harmonic = if row_context.harmonic_path_scale == 0.0 {
            1.0
        } else {
            row_context.harmonic_path_scale * path_m
        };
        self.voxel_volume_m3 * attenuation * harmonic * (row_context.k * path_m).cos() / spreading
    }
}

fn build_row_contexts(
    geometry: &HelmetHemisphereGeometry,
    receiver_indices: &[usize],
    config: &BrainHelmetFwiConfig,
) -> Vec<RowContext> {
    (0..config.measurement_count())
        .map(|row| build_row_context(row, geometry, receiver_indices, config))
        .collect()
}

fn build_row_context(
    row: usize,
    geometry: &HelmetHemisphereGeometry,
    receiver_indices: &[usize],
    config: &BrainHelmetFwiConfig,
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
        source: geometry.elements[source_idx],
        receiver: geometry.elements[receiver_idx],
        frequency_mhz: channel_frequency_hz * 1.0e-6,
        harmonic_path_scale: second_harmonic_scale(config, frequency_hz, harmonic_order),
        attenuation_model: config.attenuation_model,
        k: TAU * channel_frequency_hz / C_BRAIN_REF_M_S,
    }
}

fn second_harmonic_scale(
    config: &BrainHelmetFwiConfig,
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

fn add_partials(out: &mut [f64], partials: Vec<Vec<f64>>) {
    for partial in partials {
        for (value, increment) in out.iter_mut().zip(partial) {
            *value += increment;
        }
    }
}

fn row_chunk_len(row_count: usize) -> usize {
    let target_chunks = rayon::current_num_threads().max(1) * 4;
    row_count.div_ceil(target_chunks).max(1)
}

fn distance(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}
