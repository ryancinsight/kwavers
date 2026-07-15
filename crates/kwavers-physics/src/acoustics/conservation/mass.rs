//! Mass-continuity residual checks.

use kwavers_grid::Grid;
use leto::Array3;

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
                let div_flux = (density[[i, j, k + 1]].mul_add(
                    velocity_z[[i, j, k + 1]],
                    -(density[[i, j, k - 1]] * velocity_z[[i, j, k - 1]]),
                ) * 0.5)
                    .mul_add(
                        dz_inv,
                        (density[[i + 1, j, k]].mul_add(
                            velocity_x[[i + 1, j, k]],
                            -(density[[i - 1, j, k]] * velocity_x[[i - 1, j, k]]),
                        ) * 0.5)
                            .mul_add(
                                dx_inv,
                                density[[i, j + 1, k]].mul_add(
                                    velocity_y[[i, j + 1, k]],
                                    -(density[[i, j - 1, k]] * velocity_y[[i, j - 1, k]]),
                                ) * 0.5
                                    * dy_inv,
                            ),
                    );
                max_error = max_error.max((drho_dt + div_flux).abs());
            }
        }
    }

    max_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    use kwavers_grid::Grid;
    use leto::Array3;

    fn small_grid() -> Grid {
        Grid::new(6, 6, 6, 1e-3, 1e-3, 1e-3).unwrap()
    }

    /// Constant density, zero velocity: ∂ρ/∂t = 0 and div(ρv) = 0 → max_error = 0.
    ///
    /// This verifies the exact null case of the continuity equation.
    #[test]
    fn mass_conservation_zero_error_for_uniform_static_field() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let density = Array3::from_elem(s, DENSITY_WATER_NOMINAL);
        let density_previous = Array3::from_elem(s, DENSITY_WATER_NOMINAL);
        let velocity_x = Array3::zeros(s);
        let velocity_y = Array3::zeros(s);
        let velocity_z = Array3::zeros(s);

        let error = validate_mass_conservation(
            &density,
            &density_previous,
            &velocity_x,
            &velocity_y,
            &velocity_z,
            1e-6,
            &grid,
        );
        assert_eq!(
            error, 0.0,
            "uniform static field must give zero mass continuity error"
        );
    }

    /// Linear density gradient with zero velocity: ∂ρ/∂t=0, divergence of zero velocity=0
    /// → max_error = 0.
    #[test]
    fn mass_conservation_zero_error_for_static_gradient_field() {
        let grid = small_grid();
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut density = Array3::<f64>::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    density[[i, j, k]] = 900.0 + i as f64;
                }
            }
        }
        let density_prev = density.clone();
        let zero = Array3::<f64>::zeros((nx, ny, nz));

        let error =
            validate_mass_conservation(&density, &density_prev, &zero, &zero, &zero, 1e-6, &grid);
        assert_eq!(
            error, 0.0,
            "static gradient field with zero velocity must give zero error"
        );
    }
}
