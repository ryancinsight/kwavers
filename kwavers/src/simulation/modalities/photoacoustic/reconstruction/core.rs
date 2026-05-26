//! Image reconstruction algorithm implementations.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::Array3;
use rayon::prelude::*;
use crate::core::constants::numerical::{TWO_PI};
/// Time reversal reconstruction.
/// # Errors
/// - Returns [`KwaversError::InternalError`] if the precondition for a InternalError-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn time_reversal_reconstruction(
    grid: &Grid,
    pressure_fields: &[Array3<f64>],
    time_points: &[f64],
    speed_of_sound: f64,
    n_detectors: usize,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = grid.dimensions();
    let mut reconstructed = Array3::<f64>::zeros((nx, ny, nz));
    let detectors = compute_detector_positions(grid, n_detectors);
    let n_time = time_points.len().min(pressure_fields.len());
    if n_time == 0 {
        return Ok(reconstructed);
    }
    let t_start = time_points.first().copied().unwrap_or(0.0);
    let dt_time = if n_time >= 2 {
        (time_points[1] - time_points[0]).abs()
    } else {
        0.0
    };
    let inv_dt_time = if dt_time > 0.0 { 1.0 / dt_time } else { 0.0 };
    let mut detector_positions_m = Vec::with_capacity(detectors.len());
    for &(dx_idx, dy_idx, dz_idx) in &detectors {
        detector_positions_m.push((dx_idx * grid.dx, dy_idx * grid.dy, dz_idx * grid.dz));
    }
    let mut signals = vec![0.0f64; detectors.len() * n_time];
    for (d_idx, &(dx, dy, dz)) in detectors.iter().enumerate() {
        let base = d_idx * n_time;
        for (t_idx, field) in pressure_fields.iter().take(n_time).enumerate() {
            signals[base + t_idx] = interpolate_detector_signal(grid, field, dx, dy, dz);
        }
    }
    let nxy = ny * nz;
    let expected_len = nx * nxy;
    let out = reconstructed.as_slice_mut().ok_or_else(|| {
        KwaversError::InternalError("Reconstruction buffer not contiguous".to_owned())
    })?;
    if out.len() != expected_len {
        return Err(KwaversError::InternalError(
            "Reconstruction buffer length mismatch".to_owned(),
        ));
    }
    out.par_iter_mut().enumerate().for_each(|(idx, out_cell)| {
        let k = idx % nz;
        let j = (idx / nz) % ny;
        let i = idx / nxy;
        let px = i as f64 * grid.dx;
        let py = j as f64 * grid.dy;
        let pz = k as f64 * grid.dz;
        let mut sum = 0.0;
        for (d_idx, &(dx, dy, dz)) in detector_positions_m.iter().enumerate() {
            let rx = px - dx;
            let ry = py - dy;
            let rz = pz - dz;
            let dist = rz.mul_add(rz, rx.mul_add(rx, ry * ry)).sqrt();
            let delay = dist / speed_of_sound;
            let mut val = signals[d_idx * n_time];
            if n_time >= 2 && inv_dt_time > 0.0 {
                let pos = (delay - t_start) * inv_dt_time;
                if pos <= 0.0 {
                    val = signals[d_idx * n_time];
                } else {
                    let max_pos = (n_time - 1) as f64;
                    if pos >= max_pos {
                        val = signals[d_idx * n_time + (n_time - 1)];
                    } else {
                        let i0 = pos.floor() as usize;
                        let frac = pos - i0 as f64;
                        let base = d_idx * n_time + i0;
                        let v0 = signals[base];
                        let v1 = signals[base + 1];
                        val = v0.mul_add(1.0 - frac, v1 * frac);
                    }
                }
            }
            let weight = 1.0 / dist.max(grid.dx);
            sum += val * weight;
        }
        *out_cell = sum;
    });
    Ok(reconstructed)
}

#[must_use]
pub fn interpolate_detector_signal(
    _grid: &Grid,
    field: &Array3<f64>,
    x_det: f64,
    y_det: f64,
    z_det: f64,
) -> f64 {
    let (nx, ny, nz) = field.dim();
    let x_clamp = x_det.clamp(0.0, (nx - 1) as f64);
    let y_clamp = y_det.clamp(0.0, (ny - 1) as f64);
    let z_clamp = z_det.clamp(0.0, (nz - 1) as f64);
    let x_floor = x_clamp.floor() as usize;
    let y_floor = y_clamp.floor() as usize;
    let z_floor = z_clamp.floor() as usize;
    let x_ceil = (x_floor + 1).min(nx - 1);
    let y_ceil = (y_floor + 1).min(ny - 1);
    let z_ceil = (z_floor + 1).min(nz - 1);
    let x_weight = x_clamp - x_floor as f64;
    let y_weight = y_clamp - y_floor as f64;
    let z_weight = z_clamp - z_floor as f64;
    let c000 = field[[x_floor, y_floor, z_floor]];
    let c001 = field[[x_floor, y_floor, z_ceil]];
    let c010 = field[[x_floor, y_ceil, z_floor]];
    let c011 = field[[x_floor, y_ceil, z_ceil]];
    let c100 = field[[x_ceil, y_floor, z_floor]];
    let c101 = field[[x_ceil, y_floor, z_ceil]];
    let c110 = field[[x_ceil, y_ceil, z_floor]];
    let c111 = field[[x_ceil, y_ceil, z_ceil]];
    (c111 * x_weight * y_weight).mul_add(
        z_weight,
        (c110 * x_weight * y_weight).mul_add(
            1.0 - z_weight,
            (c101 * x_weight * (1.0 - y_weight)).mul_add(
                z_weight,
                (c100 * x_weight * (1.0 - y_weight)).mul_add(
                    1.0 - z_weight,
                    (c011 * (1.0 - x_weight) * y_weight).mul_add(
                        z_weight,
                        (c010 * (1.0 - x_weight) * y_weight).mul_add(
                            1.0 - z_weight,
                            (c000 * (1.0 - x_weight) * (1.0 - y_weight)).mul_add(
                                1.0 - z_weight,
                                c001 * (1.0 - x_weight) * (1.0 - y_weight) * z_weight,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
}

pub fn compute_detector_positions(grid: &Grid, n_detectors: usize) -> Vec<(f64, f64, f64)> {
    let (nx, ny, nz) = grid.dimensions();
    let center_x = nx as f64 / 2.0;
    let center_y = ny as f64 / 2.0;
    let center_z = nz as f64 / 2.0;
    let radius = ((nx.min(ny)) as f64 / 2.0) * 0.4;
    let mut positions = Vec::with_capacity(n_detectors);
    for i in 0..n_detectors {
        let angle = TWO_PI * i as f64 / n_detectors as f64;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        positions.push((x, y, center_z));
    }
    positions
}
