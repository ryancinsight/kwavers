//! Diagonal preconditioner for Helmholtz solvers
//!
//! This module implements a simple diagonal preconditioner that approximates
//! the inverse of the Helmholtz operator using only the diagonal elements.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::solver::forward::helmholtz::Preconditioner;
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use num_complex::Complex64;

/// Diagonal preconditioner for Helmholtz equation
#[derive(Debug)]
pub struct DiagonalPreconditioner {
    /// Diagonal elements of the preconditioner
    diagonal: Array3<Complex64>,
    /// Wavenumber for which preconditioner is set up
    wavenumber: Option<f64>,
}

impl DiagonalPreconditioner {
    /// Create a new diagonal preconditioner
    #[must_use]
    pub fn new(grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            diagonal: Array3::zeros(shape),
            wavenumber: None,
        }
    }
}

impl Preconditioner for DiagonalPreconditioner {
    /// Apply diagonal preconditioner: M⁻¹x where M is diagonal
    fn apply(
        &self,
        input: &ArrayView3<Complex64>,
        output: &mut ArrayViewMut3<Complex64>,
    ) -> KwaversResult<()> {
        Zip::from(output)
            .and(input)
            .and(&self.diagonal)
            .for_each(|out, &inp, &diag| {
                if diag.norm_sqr() > 1e-12 {
                    *out = inp / diag;
                } else {
                    *out = inp * Complex64::new(1e6, 0.0); // Fallback for near-zero diagonal
                }
            });

        Ok(())
    }

    /// Setup diagonal preconditioner for Helmholtz operator
    fn setup(&mut self, wavenumber: f64, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        let k_squared = wavenumber * wavenumber;

        // Compute diagonal elements of Helmholtz operator: -∇² - k²(1+V)
        Zip::indexed(&mut self.diagonal).for_each(|(i, j, k), diag| {
            // Approximate Laplacian diagonal contribution
            // For 3D Laplacian with second-order differences: -6/dx² per dimension
            let laplacian_diag =
                -6.0 / (grid.dx * grid.dx) - 6.0 / (grid.dy * grid.dy) - 6.0 / (grid.dz * grid.dz);

            // Get medium properties
            let c_local = medium.sound_speed(i, j, k);
            let rho_local = medium.density(i, j, k);

            // Reference values
            let c0 = 1500.0; // m/s
            let rho0 = 1000.0; // kg/m³

            // Heterogeneity potential
            let contrast = (rho_local * c_local * c_local) / (rho0 * c0 * c0);
            let heterogeneity = 1.0 - contrast;

            // Helmholtz diagonal: -∇² - k²(1+V)
            let helmholtz_diag = laplacian_diag - k_squared * heterogeneity;

            *diag = Complex64::new(helmholtz_diag, 0.0);
        });

        self.wavenumber = Some(wavenumber);
        Ok(())
    }
}
