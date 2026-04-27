//! Entropy-production checks for acoustic absorption.

use crate::domain::grid::Grid;
use ndarray::Array3;

/// Compute volumetric irreversible entropy production rate [W/K].
#[must_use]
pub fn entropy_production_rate(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
    temperature: f64,
    grid: &Grid,
) -> f64 {
    debug_assert!(temperature > 0.0, "Temperature must be positive Kelvin");
    let dv = grid.dx * grid.dy * grid.dz;
    let t0_inv = 1.0 / temperature.max(1.0);
    let mut total = 0.0_f64;
    let (nx, ny, nz) = pressure.dim();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                let alpha = absorption[[i, j, k]];
                if rho > 0.0 && c > 0.0 {
                    let p = pressure[[i, j, k]];
                    let vx = velocity_x[[i, j, k]];
                    let vy = velocity_y[[i, j, k]];
                    let vz = velocity_z[[i, j, k]];
                    let energy_density =
                        0.5 * rho * (vx * vx + vy * vy + vz * vz) + p * p / (2.0 * rho * c * c);
                    total += 2.0 * alpha * c * energy_density * t0_inv * dv;
                }
            }
        }
    }

    total
}
