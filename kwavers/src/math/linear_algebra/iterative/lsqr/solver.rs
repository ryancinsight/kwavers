//! LSQR solver implementation.
//!
//! Implements LSQR (Paige & Saunders 1982) via Lanczos bidiagonalisation
//! and Givens QR factorisation.
//!
//! ## References
//! - Paige CC, Saunders MA (1982). "LSQR: An algorithm for sparse linear equations
//!   and sparse least squares." *ACM Trans Math Software* 8(1):43–71.

use ndarray::{Array1, Array2};

use super::types::{LsqrConfig, LsqrResult, StopReason};

/// Compute L2 norm of a vector
fn norm_l2(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
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

    /// Solve the least-squares problem: minimise `‖A·x − b‖₂`.
    ///
    /// Implements LSQR (Paige & Saunders 1982) via Lanczos bidiagonalisation
    /// and Givens QR factorisation.
    pub fn solve(&self, a_matrix: &Array2<f64>, b_vector: &Array1<f64>) -> LsqrResult {
        let (_m, n) = a_matrix.dim();
        let mut x = Array1::zeros(n);

        // Initialise Lanczos bidiagonalisation
        let mut u = b_vector.clone();
        let beta = norm_l2(&u); // β₁ = ‖b‖

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

        // QR factorisation state (Paige & Saunders 1982, Table 1)
        let mut w = v.clone();
        let mut phi_bar = beta; // φ̄₁ = β₁
        let mut rho_bar = alpha; // ρ̄₁ = α₁
        let damping = self.config.damping;

        let mut residual_norms = Vec::new();
        let mut at_residual_norms = Vec::new();
        let mut rho_values = Vec::new();

        let mut stop_reason = StopReason::MaxIterations;
        let mut converged = false;

        for _iteration in 1..=self.config.max_iterations {
            // Bidiagonalisation step
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

            // Givens rotation (Paige & Saunders 1982, Table 1, step 3)
            let rho = (rho_bar.powi(2) + beta_new.powi(2) + damping.powi(2)).sqrt();
            if rho < 1e-12 {
                break;
            }

            let c = rho_bar / rho;
            let s = beta_new / rho;

            let theta_next = s * alpha_new;
            let rho_bar_next = -c * alpha_new;

            let phi = c * phi_bar;
            phi_bar *= s;

            // Solution and search-direction update
            x = x + (phi / rho) * &w;
            w = v_new.clone() - (theta_next / rho) * &w;

            rho_bar = rho_bar_next;
            rho_values.push(rho.abs());

            // Convergence estimates (Paige & Saunders 1982, §4)
            let residual_norm = phi_bar.abs();
            let at_residual_norm = phi_bar.abs() * alpha_new;

            residual_norms.push(residual_norm);
            at_residual_norms.push(at_residual_norm);

            // Stopping criteria
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

            u = u_new;
            v = v_new;
            alpha = alpha_new;
        }

        let final_residual_norm = residual_norms.last().copied().unwrap_or(beta);
        let final_at_residual_norm = at_residual_norms.last().copied().unwrap_or(alpha);
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

    /// Estimate condition number from Ritz values (diagonal elements of R)
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
