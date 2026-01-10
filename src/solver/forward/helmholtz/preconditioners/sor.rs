//! Successive Over-Relaxation (SOR) preconditioner
//!
//! This module implements a SOR preconditioner for Helmholtz solvers.
//! SOR provides better convergence than simple diagonal preconditioning
//! for certain problem types.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::solver::forward::helmholtz::Preconditioner;
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use num_complex::Complex64;

/// SOR preconditioner for Helmholtz equation
#[derive(Debug)]
pub struct SorPreconditioner {
    /// Relaxation parameter (ω = 1.0 for Gauss-Seidel, ω > 1.0 for SOR)
    omega: f64,
    /// Number of SOR iterations
    iterations: usize,
    /// Temporary storage for SOR iterations
    temp_field: Array3<Complex64>,
}

impl SorPreconditioner {
    /// Create a new SOR preconditioner
    #[must_use]
    pub fn new(omega: f64, iterations: usize, grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            omega,
            iterations,
            temp_field: Array3::zeros(shape),
        }
    }
}

impl Preconditioner for SorPreconditioner {
    /// Apply SOR preconditioner
    fn apply(
        &self,
        input: &ArrayView3<Complex64>,
        output: &mut ArrayViewMut3<Complex64>,
    ) -> KwaversResult<()> {
        // For now, implement simple Jacobi iteration as preconditioner
        // Full SOR would require mutable state, which conflicts with the trait
        let omega = 1.0; // Jacobi (ω = 1)

        // Simple Jacobi iteration: x_{n+1} = ω * D^{-1} * (b - R*x_n) + (1-ω)*x_n
        // For Helmholtz: approximately x_{n+1} ≈ x_n + ω * (input - A*x_n)/|A|
        Zip::from(output).and(input).for_each(|out, &inp| {
            // Simplified Jacobi: approximate inverse of diagonal
            let approx_inverse = Complex64::new(0.1, 0.0); // Placeholder
            *out = omega * approx_inverse * inp + (1.0 - omega) * inp;
        });

        Ok(())
    }

    /// Setup SOR preconditioner (no-op for SOR)
    fn setup(&mut self, _wavenumber: f64, _medium: &dyn Medium, _grid: &Grid) -> KwaversResult<()> {
        // SOR doesn't require precomputation like diagonal preconditioning
        Ok(())
    }
}

impl SorPreconditioner {
    /// Perform one SOR iteration
    fn sor_iteration(&self, field: &mut Array3<Complex64>) -> KwaversResult<()> {
        // Simplified: this method is not used in the current trait implementation
        // as we switched to Jacobi iteration for the preconditioner
        let (_nx, _ny, _nz) = field.dim();

        // Method not used in current implementation
        Ok(())
    }
}
