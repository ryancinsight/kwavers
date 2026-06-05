//! Time-domain sampling, migration, and objective utilities.
//!
//! [`migrate_residual`] is parallelised with Rayon: each voxel's contribution
//! is independent (no shared mutable state), so the entire NX×NY grid is
//! distributed across available cores.

use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use rayon::prelude::*;

use super::super::geometry::Point2;
use super::super::medium::PreparedTheranosticSlice;
use super::geometry::{distance, index_point_m};

const TONE_BURST_CYCLES: f64 = 2.0;

pub(super) fn stable_time_step(spacing_m: f64, shear_speed_m_s: f64) -> f64 {
    const ELASTIC_CFL: f64 = 0.30;
    ELASTIC_CFL * spacing_m / (2.0_f64.sqrt() * shear_speed_m_s)
}

pub(super) fn center_frequency(frequencies_hz: &[f64]) -> KwaversResult<f64> {
    if frequencies_hz.is_empty() {
        return Err(KwaversError::InvalidInput(
            "elastic shear reconstruction requires at least one frequency".to_owned(),
        ));
    }
    if frequencies_hz
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(KwaversError::InvalidInput(
            "elastic shear frequencies must be finite and positive".to_owned(),
        ));
    }
    Ok(frequencies_hz.iter().sum::<f64>() / frequencies_hz.len() as f64)
}

pub(super) fn time_steps(
    source_points: &[Point2],
    receiver_points: &[Point2],
    shear_speed_m_s: f64,
    frequency_hz: f64,
    dt_s: f64,
) -> usize {
    let max_distance = receiver_points
        .iter()
        .flat_map(|receiver| {
            source_points
                .iter()
                .map(|source| distance(*source, *receiver))
        })
        .fold(0.0_f64, f64::max);
    let duration_s = 2.0 * max_distance / shear_speed_m_s + TONE_BURST_CYCLES / frequency_hz;
    (duration_s / dt_s).ceil().max(32.0) as usize
}

/// Backproject receiver residuals onto the 2-D grid via time-of-flight migration.
///
/// # Algorithm
///
/// For each interior voxel `(ix, iy)` with physical coordinates `p`:
/// ```text
/// G(ix, iy) = Σ_r  E_r(t_arrival(p, src, r)) / √d_src(p)
/// ```
/// where `t_arrival = (d_src + d_rcv) / v_shear`,
/// `d_src = min_source distance(p)`, `d_rcv = distance(p, receiver_r)`,
/// and `E_r` is the windowed residual energy at the arrival sample (RMS within
/// one half-cycle).  The `1/√d_src` factor applies geometric spreading
/// compensation in 2-D.
///
/// # Parallelism
///
/// Each voxel is independent.  The flat index `ix * ny + iy` is mapped in
/// parallel via Rayon, producing a row-major `Vec<f64>` that is assembled into
/// an `Array2<f64>` with `from_shape_vec`.
pub(super) fn migrate_residual(
    prepared: &PreparedTheranosticSlice,
    residual: &Array2<f64>,
    dt_s: f64,
    center_frequency_hz: f64,
    source_points: &[Point2],
    receiver_points: &[Point2],
    shear_speed_m_s: f64,
) -> Array2<f64> {
    let dims = prepared.ct_hu.dim();
    let (nx, ny) = dims;
    let spacing_m = prepared.spacing_m;

    // Flat row-major computation: flat_idx = ix * ny + iy.
    let flat: Vec<f64> = (0..(nx * ny))
        .into_par_iter()
        .map(|flat_idx| {
            let ix = flat_idx / ny;
            let iy = flat_idx % ny;

            if !prepared.body_mask[[ix, iy]] {
                return 0.0;
            }
            let point = index_point_m(ix, iy, dims, spacing_m);
            let source_distance = source_points
                .iter()
                .map(|source| distance(point, *source))
                .fold(f64::INFINITY, f64::min)
                .max(spacing_m);
            receiver_points
                .iter()
                .enumerate()
                .map(|(row, receiver)| {
                    let receiver_distance = distance(point, *receiver).max(spacing_m);
                    let sample = (source_distance + receiver_distance) / shear_speed_m_s / dt_s;
                    let residual_value =
                        arrival_residual_energy(residual, row, sample, dt_s, center_frequency_hz);
                    residual_value / source_distance.sqrt()
                })
                .sum::<f64>()
        })
        .collect();

    Array2::from_shape_vec(dims, flat).expect("flat length equals nx * ny")
}

pub(super) fn trace_energy(data: &Array2<f64>) -> f64 {
    data.iter().map(|value| value * value).sum()
}

fn linear_sample(data: &Array2<f64>, row: usize, sample: f64) -> f64 {
    if row >= data.nrows() || sample < 0.0 {
        return 0.0;
    }
    let left = sample.floor() as usize;
    let right = left + 1;
    if right >= data.ncols() {
        return 0.0;
    }
    let alpha = sample - left as f64;
    (1.0 - alpha) * data[[row, left]] + alpha * data[[row, right]]
}

fn arrival_residual_energy(
    data: &Array2<f64>,
    row: usize,
    sample: f64,
    dt_s: f64,
    frequency_hz: f64,
) -> f64 {
    let half_width = arrival_half_window_samples(dt_s, frequency_hz);
    let sample_count = (2 * half_width + 1) as f64;
    let energy = (-(half_width as isize)..=(half_width as isize))
        .map(|offset| {
            let value = linear_sample(data, row, sample + offset as f64);
            value * value
        })
        .sum::<f64>();
    (energy / sample_count).sqrt()
}

fn arrival_half_window_samples(dt_s: f64, frequency_hz: f64) -> usize {
    (0.5 / (frequency_hz * dt_s)).ceil().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_solver::inverse::same_aperture::C_REF_M_S;
    use ndarray::Array2;

    #[test]
    fn residual_migration_samples_expected_arrival() {
        let prepared = PreparedTheranosticSlice {
            anatomy: super::super::super::config::AnatomyKind::Kidney,
            ct_hu: Array2::zeros((5, 5)),
            label: Array2::zeros((5, 5)),
            sound_speed_m_s: Array2::from_elem((5, 5), C_REF_M_S),
            attenuation_np_per_m_mhz: Array2::zeros((5, 5)),
            body_mask: Array2::from_elem((5, 5), true),
            organ_mask: Array2::from_elem((5, 5), true),
            target_mask: Array2::from_elem((5, 5), false),
            spacing_m: 1.0,
            source_slice_index: 0,
            source_dimensions: [5, 5],
            source_spacing_m: [1.0, 1.0],
            crop_bounds_index: [0, 4, 0, 4],
        };
        let source = [Point2 { x_m: 0.0, y_m: 0.0 }];
        let receiver = Point2 { x_m: 2.0, y_m: 0.0 };
        let mut residual = Array2::<f64>::zeros((1, 8));
        residual[[0, 2]] = 1.0;

        let image = migrate_residual(&prepared, &residual, 1.0, 1.0, &source, &[receiver], 1.0);

        assert!(image[[2, 2]] > image[[0, 0]]);
    }
}
