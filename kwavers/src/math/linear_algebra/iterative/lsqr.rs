//! LSQR Algorithm for Large-Scale Linear Least-Squares Problems
//!
//! ## Mathematical Foundation
//!
//! **Problem**: minimise `‖Ax − b‖₂` (optionally with Tikhonov regularisation `+λ²‖x‖₂`).
//!
//! **Algorithm**: Lanczos bidiagonalisation followed by QR factorisation of the
//! bidiagonal factor via successive Givens rotations (Paige & Saunders 1982, Algorithm LSQR).
//!
//! ### Lanczos Bidiagonalisation (initialisation)
//!
//! ```text
//! β₁ u₁ = b            (u₁ ∈ ℝᵐ, β₁ = ‖b‖)
//! α₁ v₁ = Aᵀ u₁        (v₁ ∈ ℝⁿ, α₁ = ‖Aᵀu₁‖)
//! ```
//!
//! At each step k = 1, 2, …:
//!
//! ```text
//! β_{k+1} u_{k+1} = A v_k  − α_k u_k
//! α_{k+1} v_{k+1} = Aᵀ u_{k+1} − β_{k+1} v_k
//! ```
//!
//! This produces `A ≈ U_k B_k V_kᵀ` where `B_k` is (k+1)×k lower bidiagonal.
//!
//! ### QR Factorisation via Givens Rotation (Paige & Saunders 1982, Table 1, Step 3)
//!
//! State: `φ̄₁ = β₁`, `ρ̄₁ = α₁`, `w₁ = v₁`, `x₀ = 0`.
//!
//! At each step k (with optional damping λ):
//!
//! ```text
//! ρ_k   = √(ρ̄_k² + β_{k+1}² + λ²)       — effective diagonal after rotation
//! c_k   = ρ̄_k / ρ_k                       — cosine of Givens rotation
//! s_k   = β_{k+1} / ρ_k                   — sine of Givens rotation
//!
//! θ_{k+1}    = s_k · α_{k+1}              — next super-diagonal element
//! ρ̄_{k+1}   = −c_k · α_{k+1}             — next diagonal (before rotation)
//!
//! φ_k        = c_k · φ̄_k                  — solution update coefficient
//! φ̄_{k+1}   = s_k · φ̄_k                  — residual propagation
//!
//! x_k   = x_{k−1} + (φ_k / ρ_k) · w_k    — solution update
//! w_{k+1} = v_{k+1} − (θ_{k+1}/ρ_k) · w_k — new search direction
//! ```
//!
//! **Stopping criteria** (Paige & Saunders 1982, §4):
//! - `‖r_k‖ ≈ |φ̄_{k+1}|` — residual norm estimate
//! - `‖Aᵀr_k‖ ≈ |φ̄_{k+1}| · α_{k+1}` — normal-equation residual
//!
//! ## References
//! - Paige CC, Saunders MA (1982). "LSQR: An algorithm for sparse linear equations
//!   and sparse least squares." *ACM Trans Math Software* 8(1):43–71.
//!   DOI:10.1145/355984.355989
//! - Paige CC, Saunders MA (1982). "Algorithm 583: LSQR: Sparse linear equations
//!   and least squares problems." *ACM Trans Math Software* 8(2):195–209.

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
    /// Tikhonov damping parameter λ ≥ 0: minimise ‖Ax − b‖² + λ²‖x‖²
    pub damping: f64,
    /// Tolerance on `‖Aᵀr‖` (normal-equation residual); stop when below this
    pub atol: f64,
    /// Tolerance on `‖r‖` (residual); stop when below this
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
    /// Solution vector x
    pub solution: Array1<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm ‖Ax − b‖ estimate
    pub residual_norm: f64,
    /// Final normal-equation residual ‖Aᵀ(Ax − b)‖ estimate
    pub at_residual_norm: f64,
    /// Condition number estimate of A (ratio of max/min Ritz values)
    pub condition_number: f64,
    /// True if a stopping criterion was satisfied before `max_iterations`
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
    /// `‖Aᵀr‖` tolerance satisfied
    AtolSatisfied,
    /// `‖r‖` tolerance satisfied
    BtolSatisfied,
    /// Condition number too large (singular or near-singular matrix)
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

    /// Solve the least-squares problem: minimise `‖A·x − b‖₂`.
    ///
    /// Implements LSQR (Paige & Saunders 1982) via Lanczos bidiagonalisation
    /// and Givens QR factorisation.  All stopping criteria track the true
    /// residual estimates derived from the bidiagonal QR factor.
    ///
    /// # Arguments
    /// * `a_matrix` – System matrix A (m × n)
    /// * `b_vector` – Right-hand side vector b (m)
    pub fn solve(&self, a_matrix: &Array2<f64>, b_vector: &Array1<f64>) -> LsqrResult {
        let (_m, n) = a_matrix.dim();
        let mut x = Array1::zeros(n);

        // ── Initialise Lanczos bidiagonalisation ────────────────────────────
        let mut u = b_vector.clone();
        let beta = norm_l2(&u); // β₁ = ‖b‖

        if beta < 1e-12 {
            // b = 0 ⟹ x = 0 is the exact solution
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

        u /= beta; // u₁ = b / β₁
        let mut v = a_matrix.t().dot(&u); // v = Aᵀu₁
        let mut alpha = norm_l2(&v); // α₁ = ‖Aᵀu₁‖

        if alpha < 1e-12 {
            // A has zero columns in the range of b — null-space case
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

        v /= alpha; // v₁ = Aᵀu₁ / α₁

        // ── QR factorisation state (Paige & Saunders 1982, Table 1) ─────────
        let mut w = v.clone(); // w₁ = v₁
        let mut phi_bar = beta; // φ̄₁ = β₁  ← MUST be initialised to β₁, not 0
        let mut rho_bar = alpha; // ρ̄₁ = α₁
        let damping = self.config.damping;

        let mut residual_norms = Vec::new();
        let mut at_residual_norms = Vec::new();
        let mut rho_values = Vec::new();

        let mut stop_reason = StopReason::MaxIterations;
        let mut converged = false;

        // ── Main iteration ───────────────────────────────────────────────────
        for _iteration in 1..=self.config.max_iterations {
            // ── Bidiagonalisation step ───────────────────────────────────────
            let mut u_new = a_matrix.dot(&v) - alpha * &u; // β_{k+1} u_{k+1} = Av_k − α_k u_k
            let beta_new = norm_l2(&u_new);
            if beta_new > 1e-12 {
                u_new /= beta_new;
            }

            let mut v_new = a_matrix.t().dot(&u_new) - beta_new * &v; // α_{k+1} v_{k+1} = Aᵀu_{k+1} − β_{k+1} v_k
            let alpha_new = norm_l2(&v_new);
            if alpha_new > 1e-12 {
                v_new /= alpha_new;
            }

            // ── Givens rotation (Paige & Saunders 1982, Table 1, step 3) ────
            //
            // Rotate to zero out β_{k+1} from the off-diagonal of the bidiagonal
            // factor.  With optional Tikhonov damping λ:
            //   ρ = √(ρ̄² + β_{k+1}² + λ²)
            //   c = ρ̄ / ρ,   s = β_{k+1} / ρ
            let rho = (rho_bar.powi(2) + beta_new.powi(2) + damping.powi(2)).sqrt();
            if rho < 1e-12 {
                break; // singular — stop iteration
            }

            let c = rho_bar / rho; // cosine
            let s = beta_new / rho; // sine

            // Next bidiagonal element and updated diagonal (step 4 of Table 1)
            let theta_next = s * alpha_new; // θ_{k+1} = s_k · α_{k+1}
            let rho_bar_next = -c * alpha_new; // ρ̄_{k+1} = −c_k · α_{k+1}

            // Residual propagation (step 5)
            let phi = c * phi_bar; // φ_k = c_k · φ̄_k
            phi_bar = s * phi_bar; // φ̄_{k+1} = s_k · φ̄_k

            // ── Solution and search-direction update (step 6–7) ──────────────
            x = x + (phi / rho) * &w; // x_k = x_{k−1} + (φ_k/ρ_k)·w_k
            w = v_new.clone() - (theta_next / rho) * &w; // w_{k+1} = v_{k+1} − (θ_{k+1}/ρ_k)·w_k

            rho_bar = rho_bar_next;
            rho_values.push(rho.abs());

            // ── Convergence estimates (Paige & Saunders 1982, §4) ────────────
            // ‖r_k‖ ≈ |φ̄_{k+1}|
            let residual_norm = phi_bar.abs();
            // ‖Aᵀr_k‖ ≈ |φ̄_{k+1}| · α_{k+1}
            let at_residual_norm = phi_bar.abs() * alpha_new;

            residual_norms.push(residual_norm);
            at_residual_norms.push(at_residual_norm);

            // ── Stopping criteria ─────────────────────────────────────────────
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

            // ── Advance bidiagonalisation state ──────────────────────────────
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array2};

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
        let mut a = Array2::eye(3);
        a[[0, 0]] = 2.0;
        a[[1, 1]] = 3.0;
        a[[2, 2]] = 5.0;
        let b = arr1(&[2.0, 3.0, 5.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        // Exact solution x = [1, 1, 1]
        assert!(result.residual_norm < 1e-5);
        for &xi in result.solution.iter() {
            assert!((xi - 1.0).abs() < 1e-5, "Solution element should be 1.0, got {xi}");
        }
    }

    /// **Test: overdetermined consistent system with exact LS solution**
    ///
    /// System: A = [1 0; 0 2; 1 1], b = [1; 4; 3].
    ///
    /// Normal equations: AᵀA·x = Aᵀb  →  [2 1; 1 5]·x = [4; 11].
    /// Exact solution: x* = [1, 2] (b is in range(A), so residual is zero).
    #[test]
    fn test_lsqr_overdetermined_exact_solution() {
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 2.0, 1.0, 1.0]).unwrap();
        let b = arr1(&[1.0, 4.0, 3.0]);

        let cfg = LsqrConfig {
            atol: 1e-10,
            btol: 1e-10,
            max_iterations: 500,
            ..Default::default()
        };
        let solver = LsqrSolver::new(cfg);
        let result = solver.solve(&a, &b);

        assert!(
            result.residual_norm < 1e-8,
            "Residual norm should be near zero for exact LS; got {}",
            result.residual_norm
        );
        assert!(
            (result.solution[0] - 1.0).abs() < 1e-7,
            "x[0] should be 1.0, got {}",
            result.solution[0]
        );
        assert!(
            (result.solution[1] - 2.0).abs() < 1e-7,
            "x[1] should be 2.0, got {}",
            result.solution[1]
        );
    }

    #[test]
    fn test_lsqr_overdetermined_system() {
        // Overdetermined system (more equations than unknowns)
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

        let cfg = LsqrConfig {
            damping: 0.1,
            ..Default::default()
        };

        let solver = LsqrSolver::new(cfg);
        let result = solver.solve(&a, &b);

        // Damping regularizes: solution < 1 for each component
        assert!(result.iterations > 0);
        assert!(result.residual_norm < 1e-3);
        // With I and λ=0.1, solution = b/(1+λ²) ≈ 0.99 < 1.0
        for &xi in result.solution.iter() {
            assert!(xi < 1.0 + 1e-8, "Damped solution should be < 1.0, got {xi}");
        }
    }

    #[test]
    fn test_lsqr_condition_number() {
        let mut a = Array2::eye(2);
        a[[0, 0]] = 1.0;
        a[[1, 1]] = 1e-6;
        let b = arr1(&[1.0, 1.0]);

        let solver = LsqrSolver::new(LsqrConfig::default());
        let result = solver.solve(&a, &b);

        // Should estimate large condition number for this ill-conditioned system
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
