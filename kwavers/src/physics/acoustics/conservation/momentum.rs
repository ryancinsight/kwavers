//! Linearised momentum residual checks.

use crate::domain::grid::Grid;
use ndarray::Array3;

/// Compute maximum linearised Euler residual per axis.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn validate_momentum_conservation(
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    velocity_x_previous: &Array3<f64>,
    velocity_y_previous: &Array3<f64>,
    velocity_z_previous: &Array3<f64>,
    pressure: &Array3<f64>,
    density: &Array3<f64>,
    dt: f64,
    grid: &Grid,
) -> (f64, f64, f64) {
    let mut max_err_x = 0.0_f64;
    let mut max_err_y = 0.0_f64;
    let mut max_err_z = 0.0_f64;
    let dx_inv = 1.0 / grid.dx;
    let dy_inv = 1.0 / grid.dy;
    let dz_inv = 1.0 / grid.dz;
    let dt_inv = 1.0 / dt;

    for i in 1..grid.nx - 1 {
        for j in 1..grid.ny - 1 {
            for k in 1..grid.nz - 1 {
                let rho = density[[i, j, k]];
                let dvx_dt = (velocity_x[[i, j, k]] - velocity_x_previous[[i, j, k]]) * dt_inv;
                let dpx_dx = (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) * 0.5 * dx_inv;
                max_err_x = max_err_x.max(rho.mul_add(dvx_dt, dpx_dx).abs());

                let dvy_dt = (velocity_y[[i, j, k]] - velocity_y_previous[[i, j, k]]) * dt_inv;
                let dpy_dy = (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) * 0.5 * dy_inv;
                max_err_y = max_err_y.max(rho.mul_add(dvy_dt, dpy_dy).abs());

                let dvz_dt = (velocity_z[[i, j, k]] - velocity_z_previous[[i, j, k]]) * dt_inv;
                let dpz_dz = (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) * 0.5 * dz_inv;
                max_err_z = max_err_z.max(rho.mul_add(dvz_dt, dpz_dz).abs());
            }
        }
    }

    (max_err_x, max_err_y, max_err_z)
}
