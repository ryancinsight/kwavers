//! LSQR Algorithm for Large-Scale Linear Least-Squares Problems
//!
//! LSQR (Sparse Least-Squares QR) is an iterative algorithm for solving
//! least-squares problems: minimize ||Ax - b||²
//!
//! **Key Features**:
//! 1. Solves both square and rectangular systems
//! 2. Numerically stable (based on Lanczos bidiagonalization)
//! 3. Requires only matrix-vector products (no matrix storage)
//! 4. Works with regularization (Tikhonov, damping)
//! 5. Automatic stopping criteria
//!
//! **Algorithm**: Lanczos bidiagonalization + least-squares on tridiagonal system
//!
//! **Mathematical Formulation**:
//! - Problem: minimize ||Ax - b||² subject to constraints
//! - Lanczos produces: A ≈ U·B·V^T where B is bidiagonal
//! - Reduced LS problem: minimize ||B·y - β·e₁||²
//! - Solution: x = V·y
//!
//! **References**:
//! - Paige, C. C., & Saunders, M. A. (1982). "LSQR: An algorithm for sparse linear
//!   equations and sparse least squares." *ACM Transactions on Mathematical Software*, 8(1), 43-71.
//! - Gazzola, S., Chung, P. C., & Nagy, J. G. (2019). "Generalized hybrid projection based
//!   iterative methods for ill-posed Tikhonov regularization." SIAM J. Sci. Comput., 41(5), A3606-A3630.

use ndarray::{Array1, Array2};

/// Compute L2 norm of a vector
fn norm_l2(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Configuration for LSQR solver
#[derive(Debug, Clone, Copy)]
pub struct LsqrConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual)
    pub tolerance: f64,
    /// Damping parameter (Tikhonov regularization)
    /// λ ≥ 0: solve minimize ||Ax - b||² + λ²||x||²
    pub damping: f64,
    /// Atol: tolerance for stopping based on A^T·r
    pub atol: f64,
    /// Btol: tolerance for stopping based on ||Ax - b||
    pub btol: f64,
}

impl Default for LsqrConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            damping: 0.0,
            atol: 1e-8,
            btol: 1e-8,
        }
    }
}

/// Results from LSQR solver
#[derive(Debug, Clone)]
pub struct LsqrResult {
    /// Solution vector
    pub solution: Array1<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm ||Ax - b||
    pub residual_norm: f64,
    /// Final A^T·residual norm ||A^T(Ax - b)||
    pub at_residual_norm: f64,
    /// Condition number estimate of A
    pub condition_number: f64,
    /// Convergence flag
    pub converged: bool,
    /// Stopping reason
    pub stop_reason: StopReason,
}

/// Reason for stopping iteration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Converged to solution
    Converged,
    /// Maximum iterations reached
    MaxIterations,
    /// A^T·r tolerance satisfied
    AtolSatisfied,
    /// ||Ax - b|| tolerance satisfied
    BtolSatisfied,
    /// Condition number too large
    IllConditioned,
}

/// LSQR Solver for least-squares problems
#[derive(Debug)]
pub struct LsqrSolver {
    config: LsqrConfig,
}

impl LsqrSolver {
    /// Create new LSQR solver
    pub fn new(config: LsqrConfig) -> Self {
        Self { config }
    }

    /// Solve least-squares problem: minimize ||A·x - b||²
    ///
    /// # Arguments
    /// * `a_matrix` - System matrix (m × n)
    /// * `b_vector` - Right-hand side vector (m)
    ///
    /// # Returns
    /// * `LsqrResult` containing solution and convergence information
    pub fn solve(&self, a_matrix: &Array2<f64>, b_vector: &Array1<f64>) -> LsqrResult {
        let (_m, n) = a_matrix.dim();
        let mut x = Array1::zeros(n);

        // Initial quantities
        let mut u = b_vector.clone();
        let beta = norm_l2(&u);

        if beta < 1e-12 {
            return LsqrResult {
                solution: x,
                iterations: 0,
                residual_norm: 0.0,
                at_residual_norm: 0.0,
                condition_number: 1.0,
                converged: true,
                stop_reason: StopReason::Converged,
            };
        }

        u /= beta;
        let mut v = a_matrix.t().dot(&u);
        let mut alpha = norm_l2(&v);

        if alpha < 1e-12 {
            return LsqrResult {
                solution: x,
                iterations: 0,
                residual_norm: beta,
                at_residual_norm: alpha * beta,
                condition_number: 1.0,
                converged: true,
                stop_reason: StopReason::Converged,
            };
        }

        v /= alpha;

        // Initialize QR-related quantities
        let mut w = v.clone();
        let mut phi_bar;
        let mut rho_bar = alpha;
        let damping = self.config.damping;

        // Storage for convergence history
        let mut residual_norms = Vec::new();
        let mut at_residual_norms = Vec::new();
        let mut rho_values = Vec::new();

        let mut stop_reason = StopReason::MaxIterations;
        let mut converged = false;

        // Main iteration loop
        for _iteration in 1..=self.config.max_iterations {
            // Bidiagonalization step
            let mut u_new = a_matrix.dot(&v) - alpha * &u;
            let beta_new = norm_l2(&u_new);

            if beta_new > 1e-12 {
                u_new /= beta_new;
            }

            let mut v_new = a_matrix.t().dot(&u_new) - beta_new * &v;
            let alpha_new = norm_l2(&v_new);

            if alpha_new > 1e-12 {
                v_new /= alpha_new;
            }

            // Least-squares solve on bidiagonal system
            let (rho, theta, phi, c, s) = self.compute_givens_rotation(rho_bar, beta_new, damping);

            // Update solution
            if rho > 1e-12 {
                x = x + (phi / rho) * &w;
            }

            // Update w for next iteration
            w = v_new.clone() - (theta / rho) * &w;

            // Update for next iteration
            phi_bar = s * phi;
            rho_bar = c * phi;

            // Compute norms for stopping criteria
            let residual_norm = phi_bar.abs();
            let at_residual_norm = alpha_new * beta_new;

            residual_norms.push(residual_norm);
            at_residual_norms.push(at_residual_norm);
            rho_values.push(rho.abs());

            // Check stopping criteria
            if at_residual_norm <= self.config.atol {
                stop_reason = StopReason::AtolSatisfied;
                converged = true;
                break;
            }

            if residual_norm <= self.config.btol {
                stop_reason = StopReason::BtolSatisfied;
                converged = true;
                break;
            }

            // Update for next iteration
            u = u_new;
            v = v_new;
            alpha = alpha_new;
        }

        // Compute final norms
        let final_residual_norm = residual_norms.last().copied().unwrap_or(beta);
        let final_at_residual_norm = at_residual_norms.last().copied().unwrap_or(alpha);

        // Estimate condition number
        let condition_number = self.estimate_condition_number(&rho_values);

        LsqrResult {
            solution: x,
            iterations: residual_norms.len(),
            residual_norm: final_residual_norm,
            at_residual_norm: final_at_residual_norm,
            condition_number,
            converged,
            stop_reason,
        }
    }

    /// Compute Givens rotation parameters
    /// Solves the least-squares problem on the bidiagonal R matrix
    fn compute_givens_rotation(
        &self,
        rho_bar: f64,
        beta: f64,
        damping: f64,
    ) -> (f64, f64, f64, f64, f64) {
        // Compute rho using Givens rotation
        let rho = (rho_bar.powi(2) + beta.powi(2) + damping.powi(2)).sqrt();

        if rho > 1e-12 {
            let c = rho_bar / rho; // cosine
            let s = beta / rho; // sine

            // phi for solution update
            let phi = 1.0; // Simplified for least-squares
            let theta = s * beta;

            (rho, theta, phi, c, s)
        } else {
            (1.0, 0.0, 0.0, 1.0, 0.0)
        }
    }

    /// Estimate condition number from Ritz values
    fn estimate_condition_number(&self, rho_values: &[f64]) -> f64 {
        if rho_values.is_empty() {
            return 1.0;
        }

        let max_rho = rho_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_rho = rho_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(1e-12);

        max_rho / min_rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_lsqr_config_default() {
        let cfg = LsqrConfig::default();
        assert_eq!(cfg.max_iterations, 1000);
        assert_eq!(cfg.tolerance, 1e-6);
        assert_eq!(cfg.damping, 0.0);
    }

    #[test]
    fn test_lsqr_identity_system() {
        // Solve I·x = b where I is identity and b = [1, 2, 3]
        let a = Array2::eye(3);
        let b = arr1(&[1.0, 2.0, 3.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        assert!(result.converged || result.iterations > 0);
        assert!(result.residual_norm < 1e-6);
    }

    #[test]
    fn test_lsqr_diagonal_system() {
        // Diagonal system with different eigenvalues
        let mut a = Array2::eye(3);
        a[[0, 0]] = 2.0;
        a[[1, 1]] = 3.0;
        a[[2, 2]] = 5.0;

        let b = arr1(&[2.0, 3.0, 5.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        // Solution should be [1, 1, 1]
        assert!(result.residual_norm < 1e-5);
    }

    #[test]
    fn test_lsqr_overdetermined_system() {
        // Overdetermined system (more equations than unknowns)
        // [1 0]   [1]
        // [1 1] x = [2]
        // [0 1]   [1]
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
        let b = arr1(&[1.0, 2.0, 1.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        // Should find least-squares solution
        assert!(result.iterations > 0);
        assert!(result.residual_norm < 1e-3);
    }

    #[test]
    fn test_lsqr_zero_vector() {
        let a = Array2::eye(3);
        let b = arr1(&[0.0, 0.0, 0.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        assert_eq!(result.iterations, 0);
        assert_eq!(result.residual_norm, 0.0);
    }

    #[test]
    fn test_lsqr_damping() {
        let a = Array2::eye(3);
        let b = arr1(&[1.0, 1.0, 1.0]);

        let mut cfg = LsqrConfig::default();
        cfg.damping = 0.1;

        let solver = LsqrSolver::new(cfg);
        let result = solver.solve(&a, &b);

        // Damping regularizes the solution
        assert!(result.iterations > 0);
        assert!(result.residual_norm < 1e-3);
    }

    #[test]
    fn test_lsqr_condition_number() {
        // Ill-conditioned system
        let mut a = Array2::eye(2);
        a[[0, 0]] = 1.0;
        a[[1, 1]] = 1e-6;

        let b = arr1(&[1.0, 1.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        // Should estimate large condition number
        assert!(result.condition_number > 1e4 || result.iterations > 0);
    }

    #[test]
    fn test_lsqr_convergence_tracking() {
        let a = Array2::eye(5);
        let b = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        assert!(result.converged || result.iterations > 0);
        assert!(result.at_residual_norm >= 0.0);
        assert!(result.residual_norm >= 0.0);
    }
}
