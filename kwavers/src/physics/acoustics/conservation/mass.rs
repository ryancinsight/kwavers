//! Mass-continuity residual checks.

use crate::domain::grid::Grid;
use ndarray::Array3;

/// Compute the maximum pointwise mass continuity residual.
///
/// The checked equation is `partial_t rho + div(rho v) = 0`, using centered
/// spatial differences on interior cells.
#[must_use]
pub fn validate_mass_conservation(
    density: &Array3<f64>,
    density_previous: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    dt: f64,
    grid: &Grid,
) -> f64 {
    let mut max_error = 0.0_f64;
    let dx_inv = 1.0 / grid.dx;
    let dy_inv = 1.0 / grid.dy;
    let dz_inv = 1.0 / grid.dz;
    let dt_inv = 1.0 / dt;

    for i in 1..grid.nx - 1 {
        for j in 1..grid.ny - 1 {
            for k in 1..grid.nz - 1 {
                let drho_dt = (density[[i, j, k]] - density_previous[[i, j, k]]) * dt_inv;
                let div_flux = (density[[i, j, k + 1]].mul_add(velocity_z[[i, j, k + 1]], -(density[[i, j, k - 1]] * velocity_z[[i, j, k - 1]])) * 0.5).mul_add(dz_inv, (density[[i + 1, j, k]].mul_add(velocity_x[[i + 1, j, k]], -(density[[i - 1, j, k]] * velocity_x[[i - 1, j, k]])) * 0.5).mul_add(dx_inv, density[[i, j + 1, k]].mul_add(velocity_y[[i, j + 1, k]], -(density[[i, j - 1, k]] * velocity_y[[i, j - 1, k]]))
                        * 0.5 * dy_inv));
                max_error = max_error.max((drho_dt + div_flux).abs());
            }
        }
    }

    max_error
}
