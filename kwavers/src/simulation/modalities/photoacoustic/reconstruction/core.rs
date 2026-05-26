//! Image reconstruction algorithm implementations.
//!
//! ## Time-reversal delay-and-sum (TR-DAS)
//!
//! This module provides `time_reversal_reconstruction`, a delay-and-sum
//! back-projector operating on pressure snapshots from a forward simulation.
//! It reconstructs `p₀(r)` by summing over detectors the **time-reversed**
//! signal value at the propagation delay from each grid point to each detector.
//!
//! ### Algorithm (Xu & Wang 2005, Eq. 7)
//!
//! For grid point `r` and detector `d` at position `r_d`:
//! ```text
//! p₀(r) ≈ Σ_d  w(d, r) · p(r_d, T − |r − r_d|/c)
//! ```
//! where `T` is the total recording time and `w(d, r) = 1/|r − r_d|` is the
//! spherical-spreading weight.
//!
//! The **time-reversed** index is `n_time − 1 − floor(delay / dt)`, which
//! maps the forward-recorded signal to the TR convention.
//!
//! ### References
//!
//! - Xu M, Wang LV (2005). "Universal back-projection algorithm for
//!   photoacoustic computed tomography."
//!   Phys. Rev. E 71, 016706. DOI:10.1103/PhysRevE.71.016706

use crate::core::constants::numerical::TWO_PI;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::Array3;
use rayon::prelude::*;

/// Reconstruct initial pressure `p₀(r)` from pressure snapshots via
/// time-reversed delay-and-sum back-projection.
///
/// # Arguments
/// * `grid`           – spatial grid
/// * `pressure_fields`– sequence of 3-D pressure snapshots (forward simulation)
/// * `time_points`    – physical times corresponding to each snapshot (s)
/// * `speed_of_sound` – homogeneous sound speed `c₀` (m/s)
/// * `n_detectors`    – number of virtual ring detectors
///
/// # Errors
/// Returns `Err` if the output buffer is not contiguous (should never occur
/// for freshly allocated `Array3`).
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

    let dt_time = if n_time >= 2 {
        (time_points[1] - time_points[0]).abs()
    } else {
        0.0
    };
    let inv_dt_time = if dt_time > 0.0 { 1.0 / dt_time } else { 0.0 };

    // Convert detector grid-index positions to physical coordinates (m).
    let detector_positions_m: Vec<(f64, f64, f64)> = detectors
        .iter()
        .map(|&(dx_idx, dy_idx, dz_idx)| {
            (dx_idx * grid.dx, dy_idx * grid.dy, dz_idx * grid.dz)
        })
        .collect();

    // Sample detector signals from pressure snapshots.
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
            let weight = 1.0 / dist.max(grid.dx);

            // Propagation delay in samples.
            let delay_samples = dist / (speed_of_sound * dt_time.max(f64::MIN_POSITIVE));

            // Time-reversed sample index: TR convention maps forward delay τ
            // to reversed index `n_time − 1 − round(τ/dt)`.
            // (Xu & Wang 2005, Eq. 7.)
            let val = if n_time >= 2 && inv_dt_time > 0.0 {
                let fwd_pos = delay_samples;
                if fwd_pos <= 0.0 {
                    // Zero delay → last recorded sample (fully reversed).
                    let tr_idx = n_time - 1;
                    signals[d_idx * n_time + tr_idx]
                } else if fwd_pos >= (n_time - 1) as f64 {
                    // Delay beyond recording window → first recorded sample (reversed).
                    signals[d_idx * n_time]
                } else {
                    // Linear interpolation at the time-reversed position.
                    let i0 = fwd_pos.floor() as usize;
                    let frac = fwd_pos - i0 as f64;
                    // Reversed indices: forward i0 → reversed (n_time-1-i0).
                    let tr_i0 = n_time - 1 - i0;
                    // Guard against underflow when i0 == n_time-1.
                    let tr_i1 = if i0 + 1 < n_time {
                        n_time - 1 - (i0 + 1)
                    } else {
                        0
                    };
                    let v0 = signals[d_idx * n_time + tr_i0];
                    let v1 = signals[d_idx * n_time + tr_i1];
                    // Interpolate in reversed-time direction.
                    v0.mul_add(1.0 - frac, v1 * frac)
                }
            } else {
                signals[d_idx * n_time]
            };

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
