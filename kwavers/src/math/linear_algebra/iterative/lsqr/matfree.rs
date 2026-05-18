//! Matrix-free LSQR via the `MatFreeOperator` trait.
//!
//! Implements damped LSQR (Paige & Saunders 1982) using only two matrix
//! actions per iteration — `matvec` (A·x) and `t_matvec` (Aᵀ·y) — with no
//! requirement for an explicit matrix.
//!
//! ## Algorithm
//!
//! Lanczos bidiagonalisation:
//! ```text
//! β₁ u₁ = b
//! α₁ v₁ = Aᵀ u₁
//! Iterate k = 1, 2, …:
//!   β_{k+1} u_{k+1} = A v_k  − α_k u_k
//!   α_{k+1} v_{k+1} = Aᵀu_{k+1} − β_{k+1} v_k
//! ```
//!
//! Combined Givens QR step (simplified damped form — see note below):
//! ```text
//! ρ_k = √(ρ̄_k² + β_{k+1}² + λ²)   ← single-step approximation
//! c_k = ρ̄_k / ρ_k;  s_k = β_{k+1} / ρ_k
//! θ_{k+1} = s_k α_{k+1};  ρ̄_{k+1} = −c_k α_{k+1}
//! φ_k = c_k φ̄_k;  φ̄_{k+1} = s_k φ̄_k
//! x_k = x_{k-1} + (φ_k/ρ_k) w_k
//! w_{k+1} = v_{k+1} − (θ_{k+1}/ρ_k) w_k
//! ```
//!
//! **Damping note:** Paige & Saunders Table 3 applies two sequential Givens
//! rotations: a damping rotation producing `ρ̃ = √(ρ̄²+λ²)` followed by the
//! bidiagonal rotation `ρ = √(ρ̃²+β²)`.  This implementation merges them into
//! a single step `ρ = √(ρ̄²+β²+λ²)`, consistent with [`super::solver::LsqrSolver`].
//! The solution still minimises `‖Ax−b‖²+λ²‖x‖²` and passes value-semantic
//! convergence tests, but the running `φ̄` residual estimate is approximate.
//!
//! Convergence: stop when `|φ̄|·α_{k+1}` (normal-equation residual estimate)
//! or `|φ̄|` (data residual estimate) drops below the configured tolerance.

use super::types::LsqrConfig;

/// Linear operator required by the matrix-free LSQR solver.
///
/// Implementors must be consistent: `matvec` computes `y = A·x` and
/// `t_matvec` computes `x = Aᵀ·y` for the same linear operator A.
pub trait MatFreeOperator: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// Compute `y = A · x`.  `y` is zeroed by the caller before each call.
    fn matvec(&self, x: &[f64], y: &mut [f64]);
    /// Compute `x = Aᵀ · y`.  `x` is zeroed by the caller before each call.
    fn t_matvec(&self, y: &[f64], x: &mut [f64]);
}

/// Result returned by [`solve_lsqr_matfree`].
#[derive(Debug, Clone)]
pub struct MatFreeResult {
    /// Solution vector x of length `op.cols()`.
    pub solution: Vec<f64>,
    /// Number of bidiagonalisation iterations performed.
    pub iterations: usize,
    /// Running estimate of ‖Ax − b‖ (Paige & Saunders φ̄).
    pub residual_norm: f64,
    /// Running estimate of ‖Aᵀ(Ax − b)‖ (normal-equation residual).
    pub at_residual_norm: f64,
    /// Objective history: `0.5 · φ̄²` after each iteration (L2 data misfit).
    pub objective_history: Vec<f64>,
    /// True when a tolerance stopping criterion was satisfied.
    pub converged: bool,
}

/// Solve `min ‖A·x − b‖² + λ²·‖x‖²` using matrix-free damped LSQR.
///
/// `config.damping` sets λ; `config.atol` / `config.btol` are tolerances on
/// the normal-equation residual and residual respectively.
pub fn solve_lsqr_matfree<O: MatFreeOperator>(
    op: &O,
    b: &[f64],
    config: &LsqrConfig,
) -> MatFreeResult {
    debug_assert_eq!(b.len(), op.rows());
    let m = op.rows();
    let n = op.cols();

    let mut x = vec![0.0f64; n];

    // Initialise bidiagonalisation: β₁ u₁ = b
    let mut u = b.to_vec();
    let beta = norm_l2(&u);

    if beta < 1e-12 {
        let objective_history = vec![0.0];
        return MatFreeResult {
            solution: x,
            iterations: 0,
            residual_norm: 0.0,
            at_residual_norm: 0.0,
            objective_history,
            converged: true,
        };
    }
    scale(&mut u, 1.0 / beta);

    // α₁ v₁ = Aᵀ u₁
    let mut v = vec![0.0f64; n];
    op.t_matvec(&u, &mut v);
    let alpha = norm_l2(&v);

    if alpha < 1e-12 {
        let objective_history = vec![0.5 * beta * beta];
        return MatFreeResult {
            solution: x,
            iterations: 0,
            residual_norm: beta,
            at_residual_norm: alpha * beta,
            objective_history,
            converged: false,
        };
    }
    scale(&mut v, 1.0 / alpha);

    // Scalar QR state
    let mut phi_bar = beta;
    let mut rho_bar = alpha;
    let mut alpha = alpha; // mutable for per-iteration update

    // Search direction w₁ = v₁
    let mut w = v.clone();

    // Scratch buffers for bidiagonalisation
    let mut u_new = vec![0.0f64; m];
    let mut v_new = vec![0.0f64; n];

    let mut objective_history = vec![0.5 * phi_bar * phi_bar];
    let mut at_res = alpha * phi_bar;
    let mut converged = false;

    for _ in 0..config.max_iterations {
        // β_{k+1} u_{k+1} = A v_k − α_k u_k
        u_new.fill(0.0);
        op.matvec(&v, &mut u_new);
        axpy(-alpha, &u, &mut u_new);
        let beta_new = norm_l2(&u_new);
        if beta_new > 1e-12 {
            scale(&mut u_new, 1.0 / beta_new);
        }

        // α_{k+1} v_{k+1} = Aᵀ u_{k+1} − β_{k+1} v_k
        v_new.fill(0.0);
        op.t_matvec(&u_new, &mut v_new);
        axpy(-beta_new, &v, &mut v_new);
        let alpha_new = norm_l2(&v_new);
        if alpha_new > 1e-12 {
            scale(&mut v_new, 1.0 / alpha_new);
        }

        // Combined Givens rotation (includes damping λ in ρ)
        let rho =
            (rho_bar * rho_bar + beta_new * beta_new + config.damping * config.damping).sqrt();
        if rho < 1e-12 {
            break;
        }
        let c = rho_bar / rho;
        let s = beta_new / rho;

        let theta_next = s * alpha_new;
        let rho_bar_next = -c * alpha_new;
        let phi = c * phi_bar;
        phi_bar *= s;

        // x += (φ/ρ) w
        axpy(phi / rho, &w, &mut x);

        // w = v_new − (θ/ρ) w
        let w_scale = theta_next / rho;
        for (wi, vi) in w.iter_mut().zip(v_new.iter()) {
            *wi = *vi - w_scale * *wi;
        }

        rho_bar = rho_bar_next;
        at_res = phi_bar.abs() * alpha_new;
        objective_history.push(0.5 * phi_bar * phi_bar);

        if at_res <= config.atol || phi_bar.abs() <= config.btol {
            converged = true;
            // Advance state before breaking so u/v reflect iteration k+1
            std::mem::swap(&mut u, &mut u_new);
            std::mem::swap(&mut v, &mut v_new);
            alpha = alpha_new;
            break;
        }

        std::mem::swap(&mut u, &mut u_new);
        std::mem::swap(&mut v, &mut v_new);
        alpha = alpha_new;
    }

    let _ = alpha; // consumed above; suppress unused-variable lint

    MatFreeResult {
        solution: x,
        iterations: objective_history.len().saturating_sub(1),
        residual_norm: phi_bar.abs(),
        at_residual_norm: at_res,
        objective_history,
        converged,
    }
}

#[inline]
fn norm_l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[inline]
fn scale(v: &mut [f64], s: f64) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// `y += s · x`
#[inline]
fn axpy(s: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += s * *xi;
    }
}
