//! Iterative solvers for sparse linear systems
//!
//! References:
//! - Saad (2003): "Iterative methods for sparse linear systems"

use super::csr::CompressedSparseRowMatrix;
use crate::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, ArrayView1};

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Preconditioning type
    pub preconditioner: Preconditioner,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            preconditioner: Preconditioner::None,
        }
    }
}

/// Preconditioner types
#[derive(Debug, Clone)]
pub enum Preconditioner {
    None,
    Jacobi,
    IncompleteCholesky,
}

/// Iterative solver for sparse systems
pub struct IterativeSolver {
    config: SolverConfig,
}

impl IterativeSolver {
    /// Create solver with configuration
    pub fn create(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Solve Ax = b using Conjugate Gradient method
    pub fn conjugate_gradient(
        &self,
        a: &CompressedSparseRowMatrix,
        b: ArrayView1<f64>,
        x0: Option<ArrayView1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        if a.rows != a.cols {
            return Err(KwaversError::Numerical(NumericalError::Instability {
                operation: "cg_solver".to_string(),
                condition: a.rows as f64,
            }));
        }

        let n = a.rows;
        let mut x = x0.map(|v| v.to_owned()).unwrap_or_else(|| Array1::zeros(n));

        // r = b - Ax
        let mut r = b.to_owned() - a.multiply_vector(x.view())?;
        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for iteration in 0..self.config.max_iterations {
            let ap = a.multiply_vector(p.view())?;
            let alpha = rsold / p.dot(&ap);

            x = x + alpha * &p;
            r = r - alpha * &ap;

            let rsnew = r.dot(&r);

            if rsnew.sqrt() < self.config.tolerance {
                return Ok(x);
            }

            let beta = rsnew / rsold;
            p = &r + beta * p;
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
            method: "conjugate_gradient".to_string(),
            iterations: self.config.max_iterations,
            error: rsold.sqrt(),
        }))
    }

    /// Solve using BiCGSTAB (for non-symmetric matrices)
    pub fn bicgstab(
        &self,
        a: &CompressedSparseRowMatrix,
        b: ArrayView1<f64>,
        x0: Option<ArrayView1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        let n = a.rows;
        let mut x = x0.map(|v| v.to_owned()).unwrap_or_else(|| Array1::zeros(n));

        let mut r = b.to_owned() - a.multiply_vector(x.view())?;
        let r0 = r.clone();

        let mut rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;

        let mut v = Array1::zeros(n);
        let mut p = Array1::zeros(n);

        for iteration in 0..self.config.max_iterations {
            let rho_prev = rho;
            rho = r0.dot(&r);

            if rho.abs() < 1e-14 {
                break;
            }

            let beta = (rho / rho_prev) * (alpha / omega);
            p = &r + beta * (p - omega * &v);

            v = a.multiply_vector(p.view())?;
            alpha = rho / r0.dot(&v);

            let s = &r - alpha * &v;

            if s.dot(&s).sqrt() < self.config.tolerance {
                x = x + alpha * p;
                return Ok(x);
            }

            let t = a.multiply_vector(s.view())?;
            omega = t.dot(&s) / t.dot(&t);

            x = x + alpha * &p + omega * &s;
            r = s - omega * t;

            if r.dot(&r).sqrt() < self.config.tolerance {
                return Ok(x);
            }
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "bicgstab".to_string(),
            iterations: self.config.max_iterations,
            error: r.dot(&r).sqrt(),
        }))
    }
}
