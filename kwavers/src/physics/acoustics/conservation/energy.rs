//! Acoustic energy conservation checks.

use crate::domain::grid::Grid;
use ndarray::{Array3, Zip};

/// Compute total acoustic energy and relative error against `initial_energy`.
#[must_use]
pub fn validate_energy_conservation(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    initial_energy: f64,
    grid: &Grid,
) -> f64 {
    let mut total_energy = 0.0_f64;
    let dv = grid.dx * grid.dy * grid.dz;

    Zip::from(pressure)
        .and(velocity_x)
        .and(velocity_y)
        .and(velocity_z)
        .and(density)
        .and(sound_speed)
        .for_each(|&p, &vx, &vy, &vz, &rho, &c| {
            if rho > 0.0 && c > 0.0 {
                let kinetic = 0.5 * rho * vz.mul_add(vz, vx.mul_add(vx, vy * vy));
                let potential = p * p / (2.0 * rho * c * c);
                total_energy += (kinetic + potential) * dv;
            }
        });

    (total_energy - initial_energy).abs() / initial_energy.max(1e-10)
}
