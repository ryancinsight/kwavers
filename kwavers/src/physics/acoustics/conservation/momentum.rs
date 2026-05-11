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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(6, 6, 6, 1e-3, 1e-3, 1e-3).unwrap()
    }

    /// Uniform velocity (unchanged) + uniform pressure → zero residual on all axes.
    ///
    /// dv/dt = 0 and ∇p = 0 → ρ·dv/dt + ∇p = 0 exactly.
    #[test]
    fn momentum_zero_residual_for_static_uniform_field() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let vel = Array3::from_elem(s, 0.1_f64);
        let p   = Array3::from_elem(s, 1000.0_f64);
        let rho = Array3::from_elem(s, 1000.0_f64);

        let (ex, ey, ez) = validate_momentum_conservation(
            &vel, &vel, &vel,
            &vel, &vel, &vel, // current = previous → dv/dt = 0
            &p, &rho, 1e-6, &grid,
        );

        assert_eq!(ex, 0.0, "max_err_x must be 0 for uniform static field (got {ex:.3e})");
        assert_eq!(ey, 0.0, "max_err_y must be 0 for uniform static field (got {ey:.3e})");
        assert_eq!(ez, 0.0, "max_err_z must be 0 for uniform static field (got {ez:.3e})");
    }

    /// All-zero velocity and pressure → residual exactly 0 on every axis.
    #[test]
    fn momentum_zero_residual_for_all_zero_fields() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let zero = Array3::<f64>::zeros(s);
        let rho  = Array3::from_elem(s, 1000.0_f64);

        let (ex, ey, ez) = validate_momentum_conservation(
            &zero, &zero, &zero,
            &zero, &zero, &zero,
            &zero, &rho, 1e-6, &grid,
        );

        assert_eq!(ex, 0.0);
        assert_eq!(ey, 0.0);
        assert_eq!(ez, 0.0);
    }

    /// Uniform acceleration on vx only: residual_x = ρ·ΔVx/dt, ey = ez = 0.
    ///
    /// At ρ=1000, ΔVx=0.1, dt=1e-6, uniform p (∇p=0):
    ///   residual_x = 1000 × 0.1 / 1e-6 = 1e8 [N m⁻³].
    #[test]
    fn momentum_residual_x_matches_rho_dvx_dt_for_uniform_acceleration() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let dt = 1e-6_f64;
        let delta_vx = 0.1_f64;
        let rho_val  = 1000.0_f64;
        let vx       = Array3::from_elem(s, delta_vx);
        let vx_prev  = Array3::<f64>::zeros(s);
        let vy       = Array3::<f64>::zeros(s);
        let vz       = Array3::<f64>::zeros(s);
        let p        = Array3::from_elem(s, 1000.0_f64); // uniform → ∇p = 0
        let rho      = Array3::from_elem(s, rho_val);

        let (ex, ey, ez) = validate_momentum_conservation(
            &vx, &vy, &vz,
            &vx_prev, &vy, &vz,
            &p, &rho, dt, &grid,
        );

        let expected_x = rho_val * delta_vx / dt;
        assert!(
            (ex - expected_x).abs() < 1e-6,
            "residual_x={ex:.6e}, expected {expected_x:.6e}"
        );
        assert_eq!(ey, 0.0, "ey must be 0 when only vx changes");
        assert_eq!(ez, 0.0, "ez must be 0 when only vx changes");
    }
}
