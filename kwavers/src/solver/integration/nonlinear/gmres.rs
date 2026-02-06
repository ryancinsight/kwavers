//! GMRES (Generalized Minimal Residual) Krylov Subspace Solver
//!
//! Implements GMRES iterative solver for large sparse linear systems:
//! A·x = b, where A is implicitly defined through matrix-vector products.
//!
//! Essential component of Jacobian-Free Newton-Krylov (JFNK) methods for
//! implicit multiphysics coupling without explicit Jacobian assembly.
//!
//! # Algorithm
//!
//! **GMRES with Arnoldi Orthogonalization:**
//!
//! ```text
//! Given: Linear operator A, right-hand side b, initial guess x₀
//! Goal: Minimize ||b - A·xₖ|| over Krylov subspace Kₖ(A, r₀)
//!
//! 1. r₀ = b - A·x₀,  β = ||r₀||
//! 2. v₁ = r₀ / β
//! 3. For j = 1, 2, ..., m:
//!      w = A·vⱼ
//!      For i = 1, ..., j:
//!          hᵢⱼ = ⟨w, vᵢ⟩
//!          w = w - hᵢⱼ·vᵢ  (Modified Gram-Schmidt)
//!      hⱼ₊₁,ⱼ = ||w||
//!      vⱼ₊₁ = w / hⱼ₊₁,ⱼ
//! 4. Solve least-squares: min ||β·e₁ - H̄ₘ·y||
//! 5. xₘ = x₀ + Vₘ·y
//! ```
//!
//! **Restart:** GMRES(m) restarts every m iterations to limit memory.
//!
//! # Convergence
//!
//! GMRES guarantees monotonic decrease in residual norm:
//! ```text
//! ||rₖ|| ≤ ||r₀|| · min_{p∈Pₖ} max_{λ∈σ(A)} |p(λ)|
//! ```
//! where Pₖ are polynomials with p(0) = 1.
//!
//! # References
//!
//! - Saad, Y., & Schultz, M. H. (1986). "GMRES: A generalized minimal residual
//!   algorithm for solving nonsymmetric linear systems." SIAM Journal on
//!   Scientific and Statistical Computing, 7(3), 856-869.
//!   DOI: 10.1137/0907058
//!
//! - PETSc KSP GMRES implementation:
//!   https://www.mcs.anl.gov/petsc/
//!
//! - k-Wave: Implicit pressure-velocity coupling in kspaceFirstOrder-OMP
//!   https://github.com/ucl-bug/k-wave
//!
//! - OptimUS: Iterative linear solvers for ultrasound optimization
//!   https://github.com/optimuslib/optimus

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::Array3;

/// GMRES configuration parameters
#[derive(Debug, Clone)]
pub struct GMRESConfig {
    /// Krylov subspace dimension before restart (default: 30)
    /// Typical range: 10-100
    /// Larger m → better convergence but more memory
    pub krylov_dim: usize,

    /// Maximum number of iterations (outer restarts)
    pub max_iterations: usize,

    /// Relative tolerance: ||r|| / ||b|| < tol
    pub relative_tolerance: f64,

    /// Absolute tolerance: ||r|| < tol
    pub absolute_tolerance: f64,

    /// Enable preconditioning
    pub use_preconditioner: bool,
}

impl Default for GMRESConfig {
    fn default() -> Self {
        Self {
            krylov_dim: 30,
            max_iterations: 100,
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-10,
            use_preconditioner: false,
        }
    }
}

/// GMRES solver for linear systems A·x = b
///
/// Uses restarted GMRES(m) with Modified Gram-Schmidt orthogonalization
#[derive(Debug)]
pub struct GMRESSolver {
    config: GMRESConfig,
    iteration_count: usize,
    residual_history: Vec<f64>,
}

impl GMRESSolver {
    /// Create new GMRES solver with configuration
    pub fn new(config: GMRESConfig) -> Self {
        Self {
            config,
            iteration_count: 0,
            residual_history: Vec::new(),
        }
    }

    /// Solve A·x = b using GMRES with implicit matrix-vector product
    ///
    /// # Arguments
    ///
    /// * `matvec` - Closure computing A·v for any vector v
    /// * `b` - Right-hand side vector
    /// * `x0` - Initial guess (modified in-place to solution)
    ///
    /// # Returns
    ///
    /// Convergence info with final residual norm and iteration count
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Matrix-vector product fails
    /// - Convergence not achieved within max_iterations
    /// - Numerical breakdown (zero Krylov vector)
    #[allow(non_snake_case)] // V and H are standard mathematical notation for Krylov basis and Hessenberg matrix
    pub fn solve<F>(
        &mut self,
        mut matvec: F,
        b: &Array3<f64>,
        x0: &mut Array3<f64>,
    ) -> KwaversResult<ConvergenceInfo>
    where
        F: FnMut(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        self.iteration_count = 0;
        self.residual_history.clear();

        let m = self.config.krylov_dim;

        // Initial residual: r0 = b - A·x0
        let ax0 = matvec(x0)?;
        let mut r = b - &ax0;
        let mut rho = Self::norm(&r);
        let b_norm = Self::norm(b);

        // Check if already converged
        if self.check_convergence(rho, b_norm) {
            return Ok(ConvergenceInfo {
                converged: true,
                iterations: 0,
                final_residual: rho,
                relative_residual: rho / b_norm.max(1e-15),
            });
        }

        self.residual_history.push(rho);

        // Restart loop: try multiple GMRES cycles
        for _restart_iter in 0..self.config.max_iterations {
            // Build Krylov subspace via Arnoldi
            let mut V = vec![Array3::zeros(x0.dim()); m + 1]; // Orthonormal basis vectors
            let mut H = vec![vec![0.0; m]; m + 1]; // Hessenberg matrix
            let mut gamma = vec![0.0; m + 1]; // RHS for least squares
            let mut cs = vec![0.0; m]; // Givens cosines
            let mut sn = vec![0.0; m]; // Givens sines

            // Normalize first Krylov vector
            V[0] = &r / rho;
            gamma[0] = rho;

            let mut k_steps = 0;

            // Krylov iteration
            for j in 0..m {
                // Matrix-vector product
                let w = matvec(&V[j])?;
                k_steps = j + 1;

                // Orthogonalize against previous vectors (Modified Gram-Schmidt)
                for i in 0..=j {
                    H[i][j] = Self::dot(&w, &V[i]);
                }

                // Compute new vector and check for breakdown
                let mut w_next = w.clone();
                for i in 0..=j {
                    w_next = &w_next - &(&V[i] * H[i][j]);
                }

                H[j + 1][j] = Self::norm(&w_next);

                if H[j + 1][j] < 1e-14 {
                    // Breakdown: Krylov subspace exhausted or exact solution found
                    k_steps = j + 1;
                    // Skip creating next vector, but continue to apply Givens
                } else {
                    V[j + 1] = &w_next / H[j + 1][j];
                }

                // Apply previous Givens rotations
                for i in 0..j {
                    let temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                    H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                    H[i][j] = temp;
                }

                // Generate new Givens rotation
                let (c, s) = Self::givens_rotation(H[j][j], H[j + 1][j]);
                cs[j] = c;
                sn[j] = s;

                // Apply Givens rotation to H and gamma
                H[j][j] = c * H[j][j] + s * H[j + 1][j];
                H[j + 1][j] = 0.0;
                gamma[j + 1] = -s * gamma[j];
                gamma[j] *= c;

                let residual = gamma[j + 1].abs();
                self.residual_history.push(residual);
                self.iteration_count += 1;

                // Check convergence
                if self.check_convergence(residual, b_norm) {
                    // Solve upper triangular system H_j * y = gamma_j
                    let y = Self::solve_upper_triangular(&H, &gamma, j + 1)?;

                    // Update solution
                    for i in 0..=j {
                        *x0 = &*x0 + &(&V[i] * y[i]);
                    }

                    return Ok(ConvergenceInfo {
                        converged: true,
                        iterations: self.iteration_count,
                        final_residual: residual,
                        relative_residual: residual / b_norm.max(1e-15),
                    });
                }

                // Exit if breakdown occurred
                if H[j + 1][j] < 1e-14 {
                    break;
                }
            }

            // Solve least squares problem and update x
            let y = Self::solve_upper_triangular(&H, &gamma, k_steps)?;
            for i in 0..k_steps {
                *x0 = &*x0 + &(&V[i] * y[i]);
            }

            // Compute residual for next restart
            let ax = matvec(x0)?;
            r = b - &ax;
            rho = Self::norm(&r);

            if self.check_convergence(rho, b_norm) {
                return Ok(ConvergenceInfo {
                    converged: true,
                    iterations: self.iteration_count,
                    final_residual: rho,
                    relative_residual: rho / b_norm.max(1e-15),
                });
            }

            self.residual_history.push(rho);
        }

        // Did not converge
        let final_residual = self.residual_history.last().copied().unwrap_or(rho);
        Err(KwaversError::Numerical(NumericalError::InvalidOperation(
            format!(
                "GMRES did not converge in {} iterations. Final residual: {:.3e}",
                self.iteration_count, final_residual
            ),
        )))
    }

    /// Get residual history
    pub fn residual_history(&self) -> &[f64] {
        &self.residual_history
    }

    /// Get total iteration count
    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }

    // ========== Private Methods ==========

    /// Check convergence criteria
    fn check_convergence(&self, residual: f64, b_norm: f64) -> bool {
        let relative = residual / b_norm.max(1e-15);
        residual < self.config.absolute_tolerance || relative < self.config.relative_tolerance
    }

    /// Compute L2 norm of 3D array
    fn norm(a: &Array3<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute dot product of two 3D arrays
    fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute Givens rotation (c, s) such that:
    /// [c   s] [a]   [r]
    /// [-s  c] [b] = [0]
    fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
        if b.abs() < 1e-15 {
            (1.0, 0.0)
        } else if a.abs() < b.abs() {
            let temp = a / b;
            let s = 1.0 / (1.0 + temp * temp).sqrt();
            let c = temp * s;
            (c, s)
        } else {
            let temp = b / a;
            let c = 1.0 / (1.0 + temp * temp).sqrt();
            let s = temp * c;
            (c, s)
        }
    }

    /// Solve upper triangular system H·y = g
    fn solve_upper_triangular(h: &[Vec<f64>], g: &[f64], k: usize) -> KwaversResult<Vec<f64>> {
        let mut y = vec![0.0; k];

        for i in (0..k).rev() {
            let mut sum = g[i];
            for j in i + 1..k {
                sum -= h[i][j] * y[j];
            }

            if h[i][i].abs() < 1e-15 {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    "Singular Hessenberg matrix in GMRES".to_string(),
                )));
            }

            y[i] = sum / h[i][i];
        }

        Ok(y)
    }
}

/// Convergence information from GMRES solve
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether solver converged within tolerance
    pub converged: bool,

    /// Number of iterations performed
    pub iterations: usize,

    /// Final residual norm ||b - A·x||
    pub final_residual: f64,

    /// Relative residual ||r|| / ||b||
    pub relative_residual: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gmres_identity_matrix() {
        // Solve I·x = b where I is identity
        let config = GMRESConfig {
            krylov_dim: 10,
            max_iterations: 10,
            relative_tolerance: 1e-10,
            absolute_tolerance: 1e-12,
            use_preconditioner: false,
        };

        let mut solver = GMRESSolver::new(config);

        let b = Array3::from_elem((2, 2, 2), 1.0);
        let mut x0 = Array3::zeros((2, 2, 2));

        // Identity matrix: A·v = v
        let matvec = |v: &Array3<f64>| Ok(v.clone());

        let result = solver.solve(matvec, &b, &mut x0);

        // Debug: print result
        match &result {
            Ok(info) => {
                println!(
                    "Converged: {}, iterations: {}, residual: {}",
                    info.converged, info.iterations, info.final_residual
                );
            }
            Err(e) => {
                println!("Error: {:?}", e);
                println!("Residual history len: {}", solver.residual_history().len());
                println!("Iteration count: {}", solver.iteration_count());
            }
        }

        let info = result.unwrap();

        assert!(info.converged);
        assert!(info.iterations <= 2); // Should converge in 1 iteration
        assert!(info.final_residual < 1e-10);

        // Solution should be x = b
        for (&x_val, &b_val) in x0.iter().zip(b.iter()) {
            assert_relative_eq!(x_val, b_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gmres_diagonal_matrix() {
        // Solve D·x = b where D is diagonal with entries 2.0
        let config = GMRESConfig::default();
        let mut solver = GMRESSolver::new(config);

        let b = Array3::from_elem((4, 4, 4), 4.0);
        let mut x0 = Array3::zeros((4, 4, 4));

        // Diagonal matrix: A·v = 2·v
        let matvec = |v: &Array3<f64>| Ok(v * 2.0);

        let info = solver.solve(matvec, &b, &mut x0).unwrap();

        assert!(info.converged);
        assert!(info.final_residual < 1e-6);

        // Solution should be x = b/2 = 2.0
        for &x_val in x0.iter() {
            assert_relative_eq!(x_val, 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gmres_residual_decrease() {
        // Test that residual decreases monotonically
        let config = GMRESConfig {
            krylov_dim: 10,
            max_iterations: 3,
            relative_tolerance: 1e-8,
            absolute_tolerance: 1e-10,
            use_preconditioner: false,
        };

        let mut solver = GMRESSolver::new(config);

        let b = Array3::from_elem((4, 4, 4), 1.0);
        let mut x0 = Array3::zeros((4, 4, 4));

        // Scaled identity: A·v = 1.5·v
        let matvec = |v: &Array3<f64>| Ok(v * 1.5);

        let _info = solver.solve(matvec, &b, &mut x0).unwrap();

        // Check residual history is monotonically decreasing
        let history = solver.residual_history();
        for i in 1..history.len() {
            assert!(
                history[i] <= history[i - 1] * (1.0 + 1e-10),
                "Residual increased: {} -> {}",
                history[i - 1],
                history[i]
            );
        }
    }

    #[test]
    fn test_givens_rotation() {
        let (c, s) = GMRESSolver::givens_rotation(3.0, 4.0);

        // Verify orthogonality: c² + s² = 1
        assert_relative_eq!(c * c + s * s, 1.0, epsilon = 1e-14);

        // Verify elimination: -s·a + c·b should be near zero
        let eliminated = -s * 3.0 + c * 4.0;
        assert!(eliminated.abs() < 1e-14);
    }
}
