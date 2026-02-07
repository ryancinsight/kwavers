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
///
/// Stores the inverse of the diagonal of the Helmholtz operator A = ∇² + k²,
/// computed during `setup()`. The `apply()` step performs Jacobi smoothing
/// using the stored diagonal, which is the first step toward full SOR.
#[derive(Debug)]
pub struct SorPreconditioner {
    /// Relaxation parameter (ω = 1.0 for Gauss-Seidel, ω > 1.0 for SOR)
    omega: f64,
    /// Number of SOR iterations
    #[allow(dead_code)] // Will be used when multi-iteration SOR is implemented
    iterations: usize,
    /// Inverse of the diagonal of the Helmholtz operator (precomputed in setup)
    diagonal_inv: Array3<Complex64>,
    /// Whether setup() has been called with valid operator data
    is_setup: bool,
}

impl SorPreconditioner {
    /// Create a new SOR preconditioner
    #[must_use]
    pub fn new(omega: f64, iterations: usize, grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            omega,
            iterations,
            diagonal_inv: Array3::zeros(shape),
            is_setup: false,
        }
    }
}

impl Preconditioner for SorPreconditioner {
    /// Apply SOR preconditioner
    ///
    /// Uses the precomputed diagonal inverse from `setup()`.
    /// output = ω * D⁻¹ * input (Jacobi step with relaxation)
    fn apply(
        &self,
        input: &ArrayView3<Complex64>,
        output: &mut ArrayViewMut3<Complex64>,
    ) -> KwaversResult<()> {
        if !self.is_setup {
            // Fallback: identity preconditioner if setup() hasn't been called
            output.assign(input);
            return Ok(());
        }

        let omega = self.omega;
        Zip::from(output)
            .and(input)
            .and(&self.diagonal_inv)
            .for_each(|out, &inp, &d_inv| {
                *out = omega * d_inv * inp;
            });

        Ok(())
    }

    /// Setup SOR preconditioner — precompute diagonal inverse of Helmholtz operator
    ///
    /// The diagonal of the 7-point Helmholtz stencil A = ∇² + k² is:
    ///   D(i,j,k) = -2(1/dx² + 1/dy² + 1/dz²) + k(x,y,z)²
    /// where k(x,y,z) = 2πf / c(x,y,z) is the spatially varying wavenumber.
    fn setup(&mut self, wavenumber: f64, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);
        let laplacian_diag = -2.0 * (dx2_inv + dy2_inv + dz2_inv);

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    // Compute local wavenumber from medium sound speed
                    let c = medium.sound_speed(i, j, k);
                    let local_k = if c > 0.0 {
                        wavenumber * 1480.0 / c // Scale by ratio to reference speed
                    } else {
                        wavenumber
                    };

                    // Diagonal entry: Laplacian diagonal + k²
                    let diag = Complex64::new(laplacian_diag + local_k * local_k, 0.0);

                    // Inverse with safety for near-zero diagonals
                    self.diagonal_inv[[i, j, k]] = if diag.norm() > 1e-15 {
                        Complex64::new(1.0, 0.0) / diag
                    } else {
                        Complex64::new(1.0, 0.0) // Identity for degenerate entries
                    };
                }
            }
        }

        self.is_setup = true;
        Ok(())
    }
}
