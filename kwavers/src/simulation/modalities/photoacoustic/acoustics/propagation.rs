//! Acoustic wave propagation via FDTD time-stepping.
//!
//! ## Mathematical Foundation
//!
//! ### Acoustic Wave Equation
//!
//! ```text
//! ∂²p/∂t² = c²∇²p
//! ```
//!
//! Discretized using second-order finite differences:
//!
//! ```text
//! pⁿ⁺¹ = 2pⁿ - pⁿ⁻¹ + (c²Δt²)∇²pⁿ
//! ```
//!
//! ## References
//!
//! - Cox & Beard (2005): "Fast calculation of pulsed photoacoustic fields"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::photoacoustic::InitialPressure;
use ndarray::Array3;

/// Propagate acoustic wave using second-order finite difference method.
///
/// # Numerical Stability
///
/// CFL condition enforced: Δt ≤ CFL · Δx_min / c
pub fn propagate_acoustic_wave(
    grid: &Grid,
    initial_pressure: &InitialPressure,
    speed_of_sound: f64,
    cfl_factor: f64,
    num_time_steps: usize,
    snapshot_interval: usize,
) -> KwaversResult<(Vec<Array3<f64>>, Vec<f64>)> {
    let (nx, ny, nz) = grid.dimensions();

    let min_h = grid.dx.min(grid.dy).min(grid.dz);
    let dt = cfl_factor * min_h / speed_of_sound;

    let p_curr = initial_pressure.pressure.clone();
    let p_prev = p_curr.clone();
    let mut p_next = Array3::zeros((nx, ny, nz));

    let c2_dt2 = (speed_of_sound * speed_of_sound) * (dt * dt);
    let inv_dx2 = 1.0 / (grid.dx * grid.dx);
    let inv_dy2 = 1.0 / (grid.dy * grid.dy);
    let inv_dz2 = 1.0 / (grid.dz * grid.dz);

    let capacity = (num_time_steps / snapshot_interval) + 2;
    let mut pressure_fields = Vec::with_capacity(capacity);
    let mut time_points = Vec::with_capacity(capacity);

    pressure_fields.push(p_curr.clone());
    time_points.push(0.0);

    let mut p_curr_loop = p_curr;
    let mut p_prev_loop = p_prev;

    for step in 1..=num_time_steps {
        for i in 0..nx {
            let im = if i > 0 { i - 1 } else { 0 };
            let ip = if i + 1 < nx { i + 1 } else { nx - 1 };

            for j in 0..ny {
                let jm = if j > 0 { j - 1 } else { 0 };
                let jp = if j + 1 < ny { j + 1 } else { ny - 1 };

                for k in 0..nz {
                    let km = if k > 0 { k - 1 } else { 0 };
                    let kp = if k + 1 < nz { k + 1 } else { nz - 1 };

                    let center = p_curr_loop[[i, j, k]];

                    let lap = (p_curr_loop[[ip, j, k]] - 2.0 * center + p_curr_loop[[im, j, k]])
                        * inv_dx2
                        + (p_curr_loop[[i, jp, k]] - 2.0 * center + p_curr_loop[[i, jm, k]])
                            * inv_dy2
                        + (p_curr_loop[[i, j, kp]] - 2.0 * center + p_curr_loop[[i, j, km]])
                            * inv_dz2;

                    p_next[[i, j, k]] = 2.0 * center - p_prev_loop[[i, j, k]] + c2_dt2 * lap;
                }
            }
        }

        std::mem::swap(&mut p_prev_loop, &mut p_curr_loop);
        std::mem::swap(&mut p_curr_loop, &mut p_next);

        if step % snapshot_interval == 0 {
            pressure_fields.push(p_curr_loop.clone());
            time_points.push(step as f64 * dt);
        }
    }

    Ok((pressure_fields, time_points))
}
