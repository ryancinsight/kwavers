//! Iterative solvers for sparse linear systems
//!
//! References:
//! - Saad (2003): "Iterative methods for sparse linear systems"

use super::csr::CompressedSparseRowMatrix;
use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Preconditioning type
    pub preconditioner: Preconditioner,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            preconditioner: Preconditioner::None,
            verbose: false,
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
#[derive(Debug)]
pub struct IterativeSolver {
    config: SolverConfig,
}

impl IterativeSolver {
    /// Create solver with configuration
    #[must_use]
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
        let mut x = x0.map_or_else(|| Array1::zeros(n), |v| v.to_owned());

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

    /// Solve using `BiCGSTAB` (for non-symmetric matrices)
    pub fn bicgstab(
        &self,
        a: &CompressedSparseRowMatrix,
        b: ArrayView1<f64>,
        x0: Option<ArrayView1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        let n = a.rows;
        let mut x = x0.map_or_else(|| Array1::zeros(n), |v| v.to_owned());

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
                if self.config.verbose {
                    log::info!("BiCGSTAB converged in {} iterations", iteration);
                }
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

            let residual_norm = r.dot(&r).sqrt();
            if residual_norm < self.config.tolerance {
                if self.config.verbose {
                    log::info!(
                        "BiCGSTAB converged in {} iterations, residual: {:.2e}",
                        iteration + 1,
                        residual_norm
                    );
                }
                return Ok(x);
            }
        }

        let final_residual = r.dot(&r).sqrt();
        if self.config.verbose {
            log::warn!(
                "BiCGSTAB failed to converge after {} iterations, residual: {:.2e}",
                self.config.max_iterations,
                final_residual
            );
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "bicgstab".to_string(),
            iterations: self.config.max_iterations,
            error: final_residual,
        }))
    }

    /// Solve complex system using `BiCGSTAB`
    pub fn bicgstab_complex(
        &self,
        a: &CompressedSparseRowMatrix<Complex64>,
        b: ArrayView1<Complex64>,
        x0: Option<ArrayView1<Complex64>>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = a.rows;
        let mut x = x0.map_or_else(|| Array1::from_elem(n, Complex64::default()), |v| v.to_owned());

        let mut r = b.to_owned() - a.multiply_vector(x.view())?;
        let r0 = r.clone();

        let mut rho = Complex64::new(1.0, 0.0);
        let mut alpha = Complex64::new(1.0, 0.0);
        let mut omega = Complex64::new(1.0, 0.0);

        let mut v = Array1::from_elem(n, Complex64::default());
        let mut p = Array1::from_elem(n, Complex64::default());

        for iteration in 0..self.config.max_iterations {
            let rho_prev = rho;
            rho = r0.dot(&r); // Non-conjugated dot product for BiCG orthogonality

            if rho.norm() < 1e-14 {
                if self.config.verbose {
                    log::info!("BiCGSTAB (Complex) converged/breakdown in {} iterations", iteration);
                }
                break;
            }

            let beta = (rho / rho_prev) * (alpha / omega);
            p = &r + beta * (&p - omega * &v);

            v = a.multiply_vector(p.view())?;

            let r0_v = r0.dot(&v);
            if r0_v.norm() < 1e-14 {
                 // Prevent division by zero
                 alpha = Complex64::new(1.0, 0.0);
            } else {
                 alpha = rho / r0_v;
            }

            let s = &r - alpha * &v;

            let s_norm = s.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if s_norm < self.config.tolerance {
                x = x + alpha * p;
                return Ok(x);
            }

            let t = a.multiply_vector(s.view())?;

            // Minimize residual norm: omega = (t, s) / (t, t) with conjugate inner product
            let t_norm_sqr = t.iter().map(|c| c.norm_sqr()).sum::<f64>();
            let t_s_dot = t.iter().zip(s.iter()).map(|(ti, si)| ti.conj() * si).sum::<Complex64>();

            if t_norm_sqr < 1e-14 {
                omega = Complex64::new(0.0, 0.0);
            } else {
                omega = t_s_dot / t_norm_sqr; // t_norm_sqr is real
            }

            x = x + alpha * &p + omega * &s;
            r = s - omega * t;

            let residual_norm = r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if residual_norm < self.config.tolerance {
                if self.config.verbose {
                    log::info!(
                        "BiCGSTAB (Complex) converged in {} iterations, residual: {:.2e}",
                        iteration + 1,
                        residual_norm
                    );
                }
                return Ok(x);
            }
        }

        let final_residual = r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if self.config.verbose {
            log::warn!(
                "BiCGSTAB (Complex) failed to converge after {} iterations, residual: {:.2e}",
                self.config.max_iterations,
                final_residual
            );
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "bicgstab_complex".to_string(),
            iterations: self.config.max_iterations,
            error: final_residual,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
    use num_complex::Complex64;

    #[test]
    fn test_bicgstab_complex_identity() {
        // Solve I * x = b
        let mut a = CompressedSparseRowMatrix::<Complex64>::create(2, 2);
        a.set_diagonal(0, Complex64::new(1.0, 0.0));
        a.set_diagonal(1, Complex64::new(1.0, 0.0));

        let b = Array1::from_vec(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)]);

        let config = SolverConfig::default();
        let solver = IterativeSolver::create(config);

        let x = solver.bicgstab_complex(&a, b.view(), None).unwrap();

        assert!((x[0] - Complex64::new(1.0, 1.0)).norm() < 1e-6);
        assert!((x[1] - Complex64::new(2.0, 2.0)).norm() < 1e-6);
    }

    #[test]
    fn test_bicgstab_complex_diagonal() {
        // Solve A * x = b where A = diag(2+i, 3+2i)
        // We choose values to ensure b dot b != 0 (avoid breakdown)
        let mut a = CompressedSparseRowMatrix::<Complex64>::create(2, 2);
        a.set_diagonal(0, Complex64::new(2.0, 1.0));
        a.set_diagonal(1, Complex64::new(3.0, 2.0));

        // Let target x = (1, 1)
        // b = A * x = (2+i, 3+2i)
        let b = Array1::from_vec(vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, 2.0)]);

        let config = SolverConfig::default();
        let solver = IterativeSolver::create(config);

        let x = solver.bicgstab_complex(&a, b.view(), None).unwrap();

        assert!((x[0] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
        assert!((x[1] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
    }
}
