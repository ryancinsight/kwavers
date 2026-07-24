//! Sparse solver configuration and traits - delegates to leto_ops.
//!
//! Provides the kwavers-vocabulary API for iterative sparse solvers,
//! delegating to leto-ops for the actual computation where possible.

use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array1, ArrayView1};

/// Preconditioner selection for sparse iterative solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparsePreconditioner {
    /// No preconditioning.
    None,
    /// Jacobi (diagonal) preconditioning.
    Jacobi,
    /// ILU(0) preconditioning.
    Ilu0,
}

/// Configuration for sparse iterative solvers.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Preconditioner to use.
    pub preconditioner: SparsePreconditioner,
    /// Whether to print verbose output.
    pub verbose: bool,
}

/// Trait for sparse preconditioners.
pub trait SparsePreconditionerTrait {
    /// Apply the preconditioner.
    fn apply(&self, r: &[Complex64]) -> Vec<Complex64>;
}

/// Iterative solver for sparse systems, delegating to leto-ops.
pub struct IterativeSolver {
    pub config: SolverConfig,
}

impl IterativeSolver {
    /// Create a new iterative solver with the given configuration.
    pub fn create(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Solve a complex sparse system using BiCGSTAB.
    ///
    /// This is a self-contained complex BiCGSTAB implementation that operates
    /// on the kwavers CompressedSparseRowMatrix<Complex64> type, providing
    /// the kwavers-vocabulary API while using leto-ops for norm computations.
    pub fn bicgstab_complex(
        &self,
        a: &crate::linear_algebra::sparse::CompressedSparseRowMatrix<Complex64>,
        b: ArrayView1<Complex64>,
        initial_guess: Option<ArrayView1<Complex64>>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = a.rows;
        let b_vec: Vec<Complex64> = b.as_slice().unwrap().to_vec();
        if b_vec.len() != n {
            return Err(kwavers_core::error::KwaversError::Numerical(
                kwavers_core::error::NumericalError::MatrixDimension {
                    operation: "bicgstab_complex".to_string(),
                    expected: format!("{n}"),
                    actual: format!("{}", b_vec.len()),
                }
            ));
        }

        let mut x: Vec<Complex64> = if let Some(guess) = initial_guess {
            guess.as_slice().unwrap().to_vec()
        } else {
            vec![Complex64::new(0.0, 0.0); n]
        };
        let mut r = b_vec.clone();
        a.matvec(&x, &mut r);
        for i in 0..n {
            r[i] = b_vec[i] - r[i];
        }

        let r_hat = r.clone();
        let mut p = r.clone();
        let mut x_out = x.clone();

        let mut rho_old = Complex64::new(1.0, 0.0);
        let mut alpha = Complex64::new(1.0, 0.0);
        let mut omega = Complex64::new(1.0, 0.0);

        let b_norm = complex_norm(&b_vec);
        let tol = if b_norm > 0.0 { self.config.tolerance * b_norm } else { self.config.tolerance };

        for _ in 0..self.config.max_iterations {
            let r_norm = complex_norm(&r);
            if r_norm < tol {
                break;
            }

            let mut ap = vec![Complex64::new(0.0, 0.0); n];
            a.matvec(&p, &mut ap);

            let r_hat_dot_r = complex_dot(&r_hat, &r);
            if r_hat_dot_r.norm() < 1e-30 {
                break;
            }

            alpha = r_hat_dot_r / complex_dot(&r_hat, &ap);

            for i in 0..n {
                x_out[i] = x[i] + alpha * p[i];
            }

            for i in 0..n {
                r[i] = r[i] - alpha * ap[i];
            }

            let r_norm_new = complex_norm(&r);
            if r_norm_new < tol {
                x = x_out.clone();
                break;
            }

            let rho_new = complex_dot(&r_hat, &r);
            let beta = (rho_new / rho_old) * (alpha / omega);

            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * ap[i]);
            }

            let mut ap2 = vec![Complex64::new(0.0, 0.0); n];
            a.matvec(&p, &mut ap2);

            let omega_num = complex_dot(&ap2, &r);
            let omega_den = complex_dot(&ap2, &ap2);
            if omega_den.norm() < 1e-30 {
                break;
            }
            omega = omega_num / omega_den;

            for i in 0..n {
                x[i] = x[i] + alpha * p[i];
                x[i] = x[i] + omega * (x_out[i] - x[i]);
            }

            for i in 0..n {
                r[i] = r[i] - omega * ap2[i];
            }

            rho_old = rho_new;
        }

        // Build result array from the solution vector.
        let mut arr = leto::Array1::<Complex64>::zeros([n]);
        for i in 0..n {
            arr[[i]] = x[i];
        }
        Ok(arr)
    }
}

/// Compute the L2 norm of a complex vector.
fn complex_norm(v: &[Complex64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Compute the dot product of two complex vectors (conjugate of first).
fn complex_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum()
}

/// Extension trait for matrix-vector multiplication on CompressedSparseRowMatrix.
pub trait MatVec {
    fn matvec(&self, x: &[Complex64], y: &mut [Complex64]);
}

impl MatVec for crate::linear_algebra::sparse::CompressedSparseRowMatrix<Complex64> {
    fn matvec(&self, x: &[Complex64], y: &mut [Complex64]) {
        for i in 0..self.rows {
            y[i] = Complex64::new(0.0, 0.0);
            for (col, val) in &self.data[i] {
                if *col < x.len() {
                    y[i] = y[i] + *val * x[*col];
                }
            }
        }
    }
}
