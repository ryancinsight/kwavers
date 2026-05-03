//! GMRES (Generalized Minimal Residual) Krylov subspace solver.
//!
//! ## Algorithm
//!
//! GMRES with Arnoldi orthogonalization (Modified Gram-Schmidt):
//!
//! ```text
//! Given A, b, x₀:
//! 1. r₀ = b - A·x₀,  β = ||r₀||,  v₁ = r₀/β
//! 2. For j = 1..m: w = A·vⱼ; orthogonalize vs. V₁..Vⱼ (MGS)
//! 3. Solve least-squares: min ||β·e₁ - H̄ₘ·y||
//! 4. xₘ = x₀ + Vₘ·y
//! ```
//!
//! ## References
//!
//! - Saad & Schultz (1986): SIAM JSC 7(3), 856–869. DOI: 10.1137/0907058

use super::config::GMRESConfig;
use super::types::ConvergenceInfo;
use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::Array3;

/// GMRES solver for linear systems A·x = b.
///
/// Uses restarted GMRES(m) with Modified Gram-Schmidt orthogonalization.
#[derive(Debug)]
pub struct GMRESSolver {
    pub(super) config: GMRESConfig,
    pub(super) iteration_count: usize,
    pub(super) residual_history: Vec<f64>,
}

impl GMRESSolver {
    /// Create new GMRES solver with configuration.
    pub fn new(config: GMRESConfig) -> Self {
        Self {
            config,
            iteration_count: 0,
            residual_history: Vec::new(),
        }
    }

    /// Solve A·x = b using GMRES with implicit matrix-vector product.
    #[allow(non_snake_case)]
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

        let ax0 = matvec(x0)?;
        let mut r = b - &ax0;
        let mut rho = Self::norm(&r);
        let b_norm = Self::norm(b);

        if self.check_convergence(rho, b_norm) {
            return Ok(ConvergenceInfo {
                converged: true,
                iterations: 0,
                final_residual: rho,
                relative_residual: rho / b_norm.max(1e-15),
            });
        }

        self.residual_history.push(rho);

        for _restart_iter in 0..self.config.max_iterations {
            let mut V = vec![Array3::zeros(x0.dim()); m + 1];
            let mut H = vec![vec![0.0; m]; m + 1];
            let mut gamma = vec![0.0; m + 1];
            let mut cs = vec![0.0; m];
            let mut sn = vec![0.0; m];

            V[0] = &r / rho;
            gamma[0] = rho;

            let mut k_steps = 0;

            for j in 0..m {
                let w = matvec(&V[j])?;
                k_steps = j + 1;

                for i in 0..=j {
                    H[i][j] = Self::dot(&w, &V[i]);
                }

                let mut w_next = w.clone();
                for i in 0..=j {
                    w_next = &w_next - &(&V[i] * H[i][j]);
                }

                H[j + 1][j] = Self::norm(&w_next);

                if H[j + 1][j] < 1e-14 {
                    k_steps = j + 1;
                } else {
                    V[j + 1] = &w_next / H[j + 1][j];
                }

                for i in 0..j {
                    let temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                    H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                    H[i][j] = temp;
                }

                let (c, s) = Self::givens_rotation(H[j][j], H[j + 1][j]);
                cs[j] = c;
                sn[j] = s;

                H[j][j] = c * H[j][j] + s * H[j + 1][j];
                H[j + 1][j] = 0.0;
                gamma[j + 1] = -s * gamma[j];
                gamma[j] *= c;

                let residual = gamma[j + 1].abs();
                self.residual_history.push(residual);
                self.iteration_count += 1;

                if self.check_convergence(residual, b_norm) {
                    let y = Self::solve_upper_triangular(&H, &gamma, j + 1)?;
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

                if H[j + 1][j] < 1e-14 {
                    break;
                }
            }

            let y = Self::solve_upper_triangular(&H, &gamma, k_steps)?;
            for i in 0..k_steps {
                *x0 = &*x0 + &(&V[i] * y[i]);
            }

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

        let final_residual = self.residual_history.last().copied().unwrap_or(rho);
        Err(KwaversError::Numerical(NumericalError::InvalidOperation(
            format!(
                "GMRES did not converge in {} iterations. Final residual: {:.3e}",
                self.iteration_count, final_residual
            ),
        )))
    }

    /// Get residual history.
    pub fn residual_history(&self) -> &[f64] {
        &self.residual_history
    }

    /// Get total iteration count.
    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }

    fn check_convergence(&self, residual: f64, b_norm: f64) -> bool {
        let relative = residual / b_norm.max(1e-15);
        residual < self.config.absolute_tolerance || relative < self.config.relative_tolerance
    }

    fn norm(a: &Array3<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute Givens rotation (c, s) such that [-s c][a b]ᵀ = [r 0]ᵀ.
    pub(super) fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
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
