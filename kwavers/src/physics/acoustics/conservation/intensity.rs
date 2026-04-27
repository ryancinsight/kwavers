//! Acoustic intensity and power flux.

use crate::domain::grid::Grid;
use ndarray::{Array3, Zip};

/// Compute acoustic intensity vector field `I = p v` [W/m^2].
#[must_use]
pub fn acoustic_intensity(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let ix = Zip::from(pressure)
        .and(velocity_x)
        .map_collect(|&p, &vx| p * vx);
    let iy = Zip::from(pressure)
        .and(velocity_y)
        .map_collect(|&p, &vy| p * vy);
    let iz = Zip::from(pressure)
        .and(velocity_z)
        .map_collect(|&p, &vz| p * vz);
    (ix, iy, iz)
}

/// Compute total acoustic power through a z-plane [W].
#[must_use]
pub fn acoustic_power_through_z_plane(
    pressure: &Array3<f64>,
    velocity_z: &Array3<f64>,
    k_plane: usize,
    grid: &Grid,
) -> f64 {
    let da = grid.dx * grid.dy;
    let mut power = 0.0_f64;
    let (nx, ny, _) = pressure.dim();
    for i in 0..nx {
        for j in 0..ny {
            power += pressure[[i, j, k_plane]] * velocity_z[[i, j, k_plane]] * da;
        }
    }
    power
}
