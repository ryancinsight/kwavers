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

        // r = b - A*x
        let mut r = b_vec.clone();
        let mut ax = vec![Complex64::new(0.0, 0.0); n];
        a.matvec(&x, &mut ax);
        for i in 0..n {
            r[i] = b_vec[i] - ax[i];
        }

        let r_hat = r.clone();
        // Standard BiCGSTAB initialization: p₀ = 0, v₀ = 0 so that
        // p₁ = r₀ + β*(0 − ω*0) = r₀ on the first iteration.
        let mut p = vec![Complex64::new(0.0, 0.0); n];

        let mut rho_old = Complex64::new(1.0, 0.0);
        let mut alpha = Complex64::new(1.0, 0.0);
        let mut omega = Complex64::new(1.0, 0.0);

        // Normalise tolerance against the initial residual ‖r₀‖ rather than
        // ‖b‖.  The penalty method produces ‖b‖ ≈ penalty (very large), which
        // makes a ‖b‖-relative tolerance immediately satisfied before any
        // interior DOF is updated.  Using ‖r₀‖ avoids the false-convergence
        // while still providing a meaningful stopping criterion.
        let r0_norm = complex_norm(&r);
        let tol = if r0_norm > 0.0 { self.config.tolerance * r0_norm } else { self.config.tolerance };

        for _ in 0..self.config.max_iterations {
            let r_norm = complex_norm(&r);
            if r_norm < tol {
                break;
            }

            // Compute rho = (r̂, r)
            let rho_new = complex_dot(&r_hat, &r);
            if rho_old.norm_sqr() < 1e-60 {
                break;
            }
            let beta = (rho_new / rho_old) * (alpha / omega);

            // p = r + beta * (p - omega * Ap)  [for iteration > 1]
            // On first iter rho_old=1, alpha=1, omega=1 and p = r already.
            let mut ap = vec![Complex64::new(0.0, 0.0); n];
            a.matvec(&p, &mut ap);
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * ap[i]);
            }
            a.matvec(&p, &mut ap);

            let r_hat_dot_ap = complex_dot(&r_hat, &ap);
            if r_hat_dot_ap.norm_sqr() < 1e-60 {
                break;
            }
            alpha = rho_new / r_hat_dot_ap;

            // s = r - alpha * Ap
            let mut s = r.clone();
            for i in 0..n {
                s[i] = r[i] - alpha * ap[i];
            }

            let s_norm = complex_norm(&s);
            if s_norm < tol {
                for i in 0..n {
                    x[i] = x[i] + alpha * p[i];
                }
                break;
            }

            // t = A*s
            let mut t = vec![Complex64::new(0.0, 0.0); n];
            a.matvec(&s, &mut t);

            let t_dot_t = complex_dot(&t, &t);
            if t_dot_t.norm_sqr() < 1e-60 {
                break;
            }
            omega = complex_dot(&t, &s) / t_dot_t;

            // x = x + alpha*p + omega*s
            for i in 0..n {
                x[i] = x[i] + alpha * p[i] + omega * s[i];
            }

            // r = s - omega*t
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
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
