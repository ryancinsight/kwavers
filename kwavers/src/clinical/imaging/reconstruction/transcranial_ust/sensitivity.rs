//! Finite-frequency source/receiver sensitivity assembly.

use rayon::prelude::*;
use std::f64::consts::TAU;

use super::{
    born::ActiveVoxel,
    config::{TranscranialUstBornInversionConfig, C_BRAIN_REF_M_S},
    medium::AcousticSlice,
    transducer::TranscranialBowlGeometry,
};

const C_TISSUE_DENSITY_KG_M3: f64 = 1000.0;

/// Build the row-normalized Born sensitivity matrix.
///
/// Rows are source-major, receiver-offset-major, frequency-major, and
/// harmonic-major.
/// The parallel chunk traversal preserves that deterministic row contract while
/// assigning independent rows to Rayon workers.
pub(super) fn build_sensitivity_matrix(
    medium: &AcousticSlice,
    config: &TranscranialUstBornInversionConfig,
    geometry: &TranscranialBowlGeometry,
    active: &[ActiveVoxel],
) -> Vec<f64> {
    let offset_count = config.receiver_offsets.len();
    let frequency_count = config.frequencies_hz.len();
    let harmonic_count = config.harmonic_count();
    let nrows = config.measurement_count();
    let ncols = active.len();
    let mut matrix = vec![0.0; nrows * ncols];
    let pixel_area = medium.spacing_m * medium.spacing_m;
    let receiver_indices = geometry.receiver_indices(&config.receiver_offsets);
    let attenuation_integrals = config
        .attenuation_model
        .then(|| build_element_voxel_attenuation_integrals(medium, geometry, active));

    matrix
        .par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(row, row_values)| {
            let source_idx = row / (offset_count * frequency_count * harmonic_count);
            let offset_idx = (row / (frequency_count * harmonic_count)) % offset_count;
            let frequency_idx = (row / harmonic_count) % frequency_count;
            let harmonic_idx = row % harmonic_count;
            let source = geometry.elements[source_idx];
            let receiver_idx = receiver_indices[source_idx * offset_count + offset_idx];
            let receiver = geometry.elements[receiver_idx];
            let frequency_hz = config.frequencies_hz[frequency_idx];
            let harmonic_order = harmonic_idx + 1;
            let channel_frequency_hz = harmonic_order as f64 * frequency_hz;
            let frequency_mhz = channel_frequency_hz * 1.0e-6;
            let k = TAU * channel_frequency_hz / C_BRAIN_REF_M_S;
            let mut norm2 = 0.0;

            for (col, voxel) in active.iter().enumerate() {
                let ds = distance(
                    source.x_m, source.y_m, source.z_m, voxel.x_m, voxel.y_m, voxel.z_m,
                );
                let dr = distance(
                    receiver.x_m,
                    receiver.y_m,
                    receiver.z_m,
                    voxel.x_m,
                    voxel.y_m,
                    voxel.z_m,
                );
                let spreading = (ds * dr).sqrt().max(1.0e-6);
                let attenuation = attenuation_integrals.as_ref().map_or(1.0, |integrals| {
                    let source_path = integrals[source_idx * ncols + col];
                    let receiver_path = integrals[receiver_idx * ncols + col];
                    (-(source_path + receiver_path) * frequency_mhz).exp()
                });
                let harmonic = if harmonic_order == 1 {
                    1.0
                } else {
                    second_harmonic_factor(config, frequency_hz, ds + dr)
                };
                let value = pixel_area * attenuation * harmonic * (k * (ds + dr)).cos() / spreading;
                row_values[col] = value;
                norm2 += value * value;
            }

            let norm = norm2.sqrt();
            if norm > 0.0 {
                for value in row_values {
                    *value /= norm;
                }
            }
        });
    matrix
}

fn second_harmonic_factor(
    config: &TranscranialUstBornInversionConfig,
    frequency_hz: f64,
    path_m: f64,
) -> f64 {
    let source_pressure_pa = config.source_pressure_mpa * 1.0e6;
    let omega = TAU * frequency_hz;
    let shock_distance_m = C_TISSUE_DENSITY_KG_M3 * C_BRAIN_REF_M_S.powi(3)
        / (config.nonlinear_beta * omega * source_pressure_pa);
    0.25 * (path_m / shock_distance_m).max(0.0)
}

fn build_element_voxel_attenuation_integrals(
    medium: &AcousticSlice,
    geometry: &TranscranialBowlGeometry,
    active: &[ActiveVoxel],
) -> Vec<f64> {
    let ncols = active.len();
    let mut integrals = vec![0.0; geometry.len() * ncols];
    integrals
        .par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(element_idx, row)| {
            let element = geometry.elements[element_idx];
            for (col, voxel) in active.iter().enumerate() {
                row[col] = attenuation_line_integral(
                    medium,
                    element.x_m,
                    element.y_m,
                    element.z_m,
                    voxel.x_m,
                    voxel.y_m,
                    voxel.z_m,
                );
            }
        });
    integrals
}

fn attenuation_line_integral(
    medium: &AcousticSlice,
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
) -> f64 {
    let length = distance(ax, ay, az, bx, by, bz);
    if length == 0.0 {
        return 0.0;
    }
    let steps = (length / medium.spacing_m).ceil().max(1.0) as usize;
    let ds = length / steps as f64;
    let mut integral = 0.0;
    for step in 0..steps {
        let t = (step as f64 + 0.5) / steps as f64;
        let x = ax + t * (bx - ax);
        let y = ay + t * (by - ay);
        integral += attenuation_at(medium, x, y) * ds;
    }
    integral
}

fn attenuation_at(medium: &AcousticSlice, x_m: f64, y_m: f64) -> f64 {
    let (nx, ny) = medium.attenuation_np_per_m_mhz.dim();
    let ix = x_m / medium.spacing_m + (nx - 1) as f64 / 2.0;
    let iy = y_m / medium.spacing_m + (ny - 1) as f64 / 2.0;
    if ix < 0.0 || iy < 0.0 || ix > (nx - 1) as f64 || iy > (ny - 1) as f64 {
        return 0.0;
    }
    let x0 = ix.floor() as usize;
    let y0 = iy.floor() as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let tx = ix - x0 as f64;
    let ty = iy - y0 as f64;
    let a00 = medium.attenuation_np_per_m_mhz[[x0, y0]];
    let a10 = medium.attenuation_np_per_m_mhz[[x1, y0]];
    let a01 = medium.attenuation_np_per_m_mhz[[x0, y1]];
    let a11 = medium.attenuation_np_per_m_mhz[[x1, y1]];
    (1.0 - tx) * (1.0 - ty) * a00 + tx * (1.0 - ty) * a10 + (1.0 - tx) * ty * a01 + tx * ty * a11
}

fn distance(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}
