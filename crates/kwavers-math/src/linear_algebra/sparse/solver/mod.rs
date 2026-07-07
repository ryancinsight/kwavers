//! Iterative solvers for sparse linear systems.
//!
//! References:
//! - Saad (2003): "Iterative methods for sparse linear systems"

mod bicgstab;

#[cfg(test)]
mod tests;

use super::csr::CompressedSparseRowMatrix;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::Array1;
use num_complex::Complex64;

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

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn scale_inplace(x: &mut Array1<f64>, a: f64) {
    for xi in x.iter_mut() {
        *xi *= a;
    }
}

fn axpy(a: f64, x: &Array1<f64>, y: &mut Array1<f64>) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * *xi;
    }
}

fn conj_dot_complex(a: &Array1<Complex64>, b: &Array1<Complex64>) -> Complex64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.conj() * y)
        .sum()
}

fn norm_sqr_complex(a: &Array1<Complex64>) -> f64 {
    a.iter().map(|c| c.norm_sqr()).sum()
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
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn conjugate_gradient(
        &self,
        a: &CompressedSparseRowMatrix,
        b: &Array1<f64>,
        x0: Option<&Array1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        if a.rows != a.cols {
            return Err(KwaversError::Numerical(NumericalError::Instability {
                operation: "cg_solver".to_owned(),
                condition: a.rows as f64,
            }));
        }

        let n = a.rows;
        let mut x = x0.map_or_else(|| Array1::zeros([n]), |v| v.clone());

        let mut r = {
            let mut tmp = b.clone();
            let ax = a.multiply_vector(&x)?;
            for i in 0..n {
                tmp[[i]] -= ax[[i]];
            }
            tmp
        };
        let mut p = r.clone();
        let mut rsold = dot(&r, &r);

        for _iteration in 0..self.config.max_iterations {
            let ap = a.multiply_vector(&p)?;
            let alpha = rsold / dot(&p, &ap);

            axpy(alpha, &p, &mut x);
            axpy(-alpha, &ap, &mut r);

            let rsnew = dot(&r, &r);

            if rsnew.sqrt() < self.config.tolerance {
                return Ok(x);
            }

            let beta = rsnew / rsold;
            scale_inplace(&mut p, beta);
            axpy(1.0, &r, &mut p);
            rsold = rsnew;
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "conjugate_gradient".to_owned(),
            iterations: self.config.max_iterations,
            error: rsold.sqrt(),
        }))
    }

    /// Solve using BiCGSTAB for non-symmetric matrices.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn bicgstab(
        &self,
        a: &CompressedSparseRowMatrix,
        b: &Array1<f64>,
        x0: Option<&Array1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        self.bicgstab_real(a, b, x0)
    }

    /// Solve complex system using BiCGSTAB.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn bicgstab_complex(
        &self,
        a: &CompressedSparseRowMatrix<Complex64>,
        b: &Array1<Complex64>,
        x0: Option<&Array1<Complex64>>,
    ) -> KwaversResult<Array1<Complex64>> {
        self.bicgstab_complex_impl(a, b, x0)
    }
}
