use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;

use super::super::wave_model::NonlinearWave;

impl NonlinearWave {
    /// Computes the nonlinear term for the acoustic wave equation.
    ///
    /// The nonlinear term accounts for finite-amplitude effects in acoustic propagation.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current pressure field \[Pa\]
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// The nonlinear term contribution
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn compute_nonlinear_term(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Get position-dependent medium properties for heterogeneous media
        // This implementation properly handles spatial variation in properties
        //
        // References:
        // - Hamilton & Blackstock (1998): "Nonlinear Acoustics" - heterogeneous nonlinearity
        // - Varslot & Taraldsen (2005): "Computer simulation of forward wave propagation"

        let (nx, ny, nz) = pressure.dim();
        let mut nonlinear_term = Array3::zeros((nx, ny, nz));

        // Compute pressure gradients using spectral differentiation
        let (grad_x, grad_y, grad_z) = self.compute_spectral_gradient(pressure, grid)?;

        // Compute Laplacian for p∇²p term
        let laplacian = self.compute_spectral_laplacian(pressure, grid)?;

        // For each grid point, compute spatially-varying nonlinear contribution
        // This properly accounts for heterogeneous media
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    // Get local medium properties
                    let density = crate::domain::medium::density_at(medium, x, y, z, grid);
                    let sound_speed = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);

                    // Get nonlinearity parameter B/A (default to water if not available)
                    // Future: Add B/A to Medium trait for full heterogeneous support
                    let nonlinearity = 3.5; // B/A for water

                    // Nonlinearity parameter: β = 1 + B/(2A)
                    let beta = 1.0 + nonlinearity / 2.0;

                    // Prefactor for Westervelt equation
                    let prefactor = beta / (density * sound_speed.powi(4));

                    // Compute nonlinear term: N = (β/ρ₀c₀⁴) * [p∇²p + (∇p)²]
                    let p_lap = pressure[[i, j, k]] * laplacian[[i, j, k]];
                    let grad_squared = grad_z[[i, j, k]].mul_add(grad_z[[i, j, k]], grad_y[[i, j, k]].mul_add(grad_y[[i, j, k]], grad_x[[i, j, k]].powi(2)));

                    nonlinear_term[[i, j, k]] = prefactor * (p_lap + grad_squared);
                }
            }
        }

        Ok(nonlinear_term)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::wave_model::NonlinearWave;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use ndarray::Array3;

    /// A spatially uniform (constant) pressure field has zero gradient and zero
    /// Laplacian. Both terms of the Westervelt nonlinear operator vanish, so the
    /// nonlinear contribution must be identically zero everywhere.
    ///
    /// Tolerance: N·ε_mach·10 where N = nx·ny·nz = 512 for an 8³ grid.
    #[test]
    fn compute_nonlinear_term_zero_for_constant_pressure_field() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);

        // Constant field: gradient = 0, Laplacian = 0 → nonlinear term = 0
        let pressure = Array3::<f64>::from_elem((8, 8, 8), 1_000.0);

        let term = w.compute_nonlinear_term(&pressure, &medium, &grid).unwrap();

        let tol = 512.0 * f64::EPSILON * 10.0;
        for &v in term.iter() {
            assert!(
                v.abs() < tol,
                "nonlinear term must be zero for constant pressure (got {v:.3e}, tol {tol:.3e})"
            );
        }
    }

    /// Zero pressure field → zero nonlinear term (trivial null case).
    #[test]
    fn compute_nonlinear_term_zero_for_zero_pressure_field() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let w = NonlinearWave::new(&grid, 1e-7);

        let pressure = Array3::<f64>::zeros((8, 8, 8));
        let term = w.compute_nonlinear_term(&pressure, &medium, &grid).unwrap();

        for &v in term.iter() {
            assert_eq!(v, 0.0, "zero pressure must give zero nonlinear term");
        }
    }
}
