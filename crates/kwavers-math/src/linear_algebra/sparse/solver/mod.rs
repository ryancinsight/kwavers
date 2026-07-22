//! Iterative solvers for sparse linear systems.
//!
//! References:
//! - Saad (2003): "Iterative methods for sparse linear systems"

mod bicgstab;

#[cfg(test)]
mod tests;

use super::csr::CompressedSparseRowMatrix;
use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{Array1, ArrayView1};

pub(super) fn dot_real(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Preconditioning type.
    pub preconditioner: SparsePreconditioner,
    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            preconditioner: SparsePreconditioner::None,
            verbose: false,
        }
    }
}

/// SparsePreconditioner types.
#[derive(Debug, Clone)]
pub enum SparsePreconditioner {
    None,
    Jacobi,
    IncompleteCholesky,
}

/// Iterative solver for sparse systems.
#[derive(Debug)]
pub struct IterativeSolver {
    pub(super) config: SolverConfig,
}

impl IterativeSolver {
    /// Create solver with configuration.
    #[must_use]
    pub fn create(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Solve Ax = b using Conjugate Gradient method (symmetric positive-definite A).
    ///
    /// Implements CG per Hestenes & Stiefel (1952).
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if A is not square.
    /// - Returns [`KwaversError::Numerical`] if the method fails to converge within `max_iterations`.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn conjugate_gradient(
        &self,
        a: &CompressedSparseRowMatrix,
        b: ArrayView1<f64>,
        x0: Option<ArrayView1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        if a.rows != a.cols {
            return Err(KwaversError::Numerical(NumericalError::Instability {
                operation: "cg_solver".to_owned(),
                condition: a.rows as f64,
            }));
        }

        let n = a.rows;
        let mut x = if let Some(v) = x0 {
            Array1::from_shape_vec([n], v.iter().copied().collect())
                .expect("initial guess length must match solver dimension")
        } else {
            Array1::zeros([n])
        };

        let b_owned = Array1::from_shape_vec([n], b.iter().copied().collect())
            .expect("rhs length must match solver dimension");
        let mut r = &b_owned - &a.multiply_vector(x.view())?;
        let mut p = r.clone();
        let mut rsold = dot_real(&r, &r);

        for iteration in 0..self.config.max_iterations {
            let ap = a.multiply_vector(p.view())?;
            let alpha = rsold / dot_real(&p, &ap);

            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);

            let rsnew = dot_real(&r, &r);

            if rsnew.sqrt() < self.config.tolerance {
                return Ok(x);
            }

            let beta = rsnew / rsold;
            p = &r + &(&p * beta);
            rsold = rsnew;

            if iteration % 50 == 0 {
                log::debug!(
                    "CG iteration {}: residual = {:.2e}",
                    iteration,
                    rsnew.sqrt()
                );
            }
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "conjugate_gradient".to_owned(),
            iterations: self.config.max_iterations,
            error: rsold.sqrt(),
        }))
    }

    /// Solve using BiCGSTAB for non-symmetric matrices.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn bicgstab(
        &self,
        a: &CompressedSparseRowMatrix,
        b: ArrayView1<f64>,
        x0: Option<ArrayView1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        self.bicgstab_real(a, b, x0)
    }

    /// Solve complex system using BiCGSTAB.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn bicgstab_complex(
        &self,
        a: &CompressedSparseRowMatrix<Complex64>,
        b: ArrayView1<Complex64>,
        x0: Option<ArrayView1<Complex64>>,
    ) -> KwaversResult<Array1<Complex64>> {
        self.bicgstab_complex_impl(a, b, x0)
    }
}
