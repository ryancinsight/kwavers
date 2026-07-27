//! GMRES (Generalized Minimal Residual) Krylov subspace solver.
//!
//! ## Algorithm
//!
//! Restarted GMRES(m) with Arnoldi orthogonalization (Modified Gram-Schmidt):
//!
//! ```text
//! Given A, b, x₀:
//! 1. r₀ = b - A·x₀,  β = ||r₀||,  v₁ = r₀/β
//! 2. For j = 1..m: w = A·vⱼ; orthogonalize vs. V₁..Vⱼ (MGS)
//! 3. Solve least-squares: min ||β·e₁ - H̄ₘ·y||
//! 4. xₘ = x₀ + Vₘ·y
//! ```
//!
//! The operator is applied matrix-free through a caller-supplied closure, so
//! the Krylov basis holds full 3-D fields. Every basis vector and scratch
//! field is therefore retained in [`Workspace`] across solves: a Newton loop
//! calls this solver once per outer iteration, and reallocating `m + 1` fields
//! per restart dominated the solve for the default `krylov_dim = 30`.
//!
//! ## References
//!
//! - Saad & Schultz (1986): SIAM JSC 7(3), 856–869. DOI: 10.1137/0907058
//! - Saad (2003): *Iterative Methods for Sparse Linear Systems*, 2nd ed., §6.5.

use super::config::GMRESConfig;
use super::types::GmresConvergenceInfo;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::Array3;

/// Retained scratch space for one `(shape, krylov_dim)` pair.
#[derive(Debug)]
struct Workspace {
    /// `krylov_dim + 1` orthonormal basis fields.
    basis: Vec<Array3<f64>>,
    /// Operator image of the current basis vector, orthogonalised in place.
    work: Array3<f64>,
    /// Current residual `b - A·x`.
    residual: Array3<f64>,
    /// Hessenberg matrix, `H[row][column]`.
    hessenberg: Vec<Vec<f64>>,
    /// Rotated right-hand side.
    gamma: Vec<f64>,
    cosines: Vec<f64>,
    sines: Vec<f64>,
}

impl Workspace {
    fn new(shape: [usize; 3], krylov_dim: usize) -> Self {
        Self {
            basis: vec![Array3::zeros(shape); krylov_dim + 1],
            work: Array3::zeros(shape),
            residual: Array3::zeros(shape),
            hessenberg: vec![vec![0.0; krylov_dim]; krylov_dim + 1],
            gamma: vec![0.0; krylov_dim + 1],
            cosines: vec![0.0; krylov_dim],
            sines: vec![0.0; krylov_dim],
        }
    }

    fn matches(&self, shape: [usize; 3], krylov_dim: usize) -> bool {
        self.basis.len() == krylov_dim + 1 && self.work.shape() == shape
    }

    /// Clear the rotation state at the start of a restart cycle. The basis and
    /// operator scratch are fully overwritten before they are read.
    fn reset_cycle(&mut self) {
        for row in &mut self.hessenberg {
            row.fill(0.0);
        }
        self.gamma.fill(0.0);
        self.cosines.fill(0.0);
        self.sines.fill(0.0);
    }
}

/// GMRES solver for linear systems A·x = b.
///
/// Uses restarted GMRES(m) with Modified Gram-Schmidt orthogonalization.
#[derive(Debug)]
pub struct GMRESSolver {
    pub(super) config: GMRESConfig,
    pub(super) iteration_count: usize,
    pub(super) residual_history: Vec<f64>,
    workspace: Option<Workspace>,
}

impl GMRESSolver {
    /// Create new GMRES solver with configuration.
    #[must_use]
    pub fn new(config: GMRESConfig) -> Self {
        Self {
            config,
            iteration_count: 0,
            residual_history: Vec::new(),
            workspace: None,
        }
    }

    /// Solve A·x = b using GMRES with implicit matrix-vector product.
    ///
    /// # Errors
    /// - [`NumericalError::InvalidOperation`] if the recurrence produces a
    ///   non-finite value, if the Krylov subspace is exhausted before the
    ///   tolerance is met, or if the iteration budget runs out.
    /// - Propagates any [`KwaversError`] returned by `matvec`.
    pub fn solve<F>(
        &mut self,
        mut matvec: F,
        b: &Array3<f64>,
        x0: &mut Array3<f64>,
    ) -> KwaversResult<GmresConvergenceInfo>
    where
        F: FnMut(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        self.iteration_count = 0;
        self.residual_history.clear();

        let m = self.config.krylov_dim;
        if m == 0 {
            return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                "GMRES krylov_dim must be positive".to_owned(),
            )));
        }

        let shape = x0.shape();
        if self
            .workspace
            .as_ref()
            .is_none_or(|workspace| !workspace.matches(shape, m))
        {
            self.workspace = Some(Workspace::new(shape, m));
        }
        let Some(workspace) = self.workspace.as_mut() else {
            unreachable!("invariant: the workspace was just installed")
        };

        let b_norm = norm(b);
        let ax0 = matvec(x0)?;
        assign_difference(&mut workspace.residual, b, &ax0);
        let mut rho = norm(&workspace.residual);
        check_finite(rho, "initial residual")?;

        if check_convergence(&self.config, rho, b_norm) {
            return Ok(converged_info(0, rho, b_norm));
        }
        self.residual_history.push(rho);

        for _restart in 0..self.config.max_iterations {
            workspace.reset_cycle();
            scale_into(&mut workspace.basis[0], &workspace.residual, 1.0 / rho);
            workspace.gamma[0] = rho;

            let mut vectors_used = 0usize;
            let mut breakdown = false;

            for j in 0..m {
                let image = matvec(&workspace.basis[j])?;
                workspace.work.assign(&image);
                let reference_norm = norm(&workspace.work);
                check_finite(reference_norm, "operator image")?;

                // Modified Gram-Schmidt: subtract each projection before the
                // next coefficient is formed. Computing every coefficient
                // against the original image first is the classical variant,
                // whose loss of orthogonality is second order in the machine
                // epsilon rather than first order.
                for i in 0..=j {
                    let coefficient = dot(&workspace.work, &workspace.basis[i]);
                    check_finite(coefficient, "Hessenberg entry")?;
                    workspace.hessenberg[i][j] = coefficient;
                    axpy(&mut workspace.work, -coefficient, &workspace.basis[i]);
                }

                let next_norm = norm(&workspace.work);
                check_finite(next_norm, "Arnoldi subdiagonal")?;
                workspace.hessenberg[j + 1][j] = next_norm;
                vectors_used = j + 1;

                // Breakdown is measured against the pre-orthogonalisation norm:
                // the subtractions above cancel at most ‖A·vⱼ‖ of magnitude, so
                // a remainder at or below ε·‖A·vⱼ‖ carries no significant
                // digits. An absolute threshold would be scale-dependent.
                if next_norm <= f64::EPSILON * reference_norm {
                    breakdown = true;
                } else {
                    let inverse = 1.0 / next_norm;
                    scale_into(&mut workspace.basis[j + 1], &workspace.work, inverse);
                }

                for i in 0..j {
                    let upper = workspace.hessenberg[i][j];
                    let lower = workspace.hessenberg[i + 1][j];
                    workspace.hessenberg[i][j] =
                        workspace.cosines[i].mul_add(upper, workspace.sines[i] * lower);
                    workspace.hessenberg[i + 1][j] =
                        (-workspace.sines[i]).mul_add(upper, workspace.cosines[i] * lower);
                }

                let (cosine, sine) = Self::givens_rotation(
                    workspace.hessenberg[j][j],
                    workspace.hessenberg[j + 1][j],
                );
                workspace.cosines[j] = cosine;
                workspace.sines[j] = sine;
                workspace.hessenberg[j][j] = cosine.mul_add(
                    workspace.hessenberg[j][j],
                    sine * workspace.hessenberg[j + 1][j],
                );
                workspace.hessenberg[j + 1][j] = 0.0;
                workspace.gamma[j + 1] = -sine * workspace.gamma[j];
                workspace.gamma[j] *= cosine;

                let estimate = workspace.gamma[j + 1].abs();
                check_finite(estimate, "residual estimate")?;
                self.residual_history.push(estimate);
                self.iteration_count += 1;

                if check_convergence(&self.config, estimate, b_norm) || breakdown {
                    break;
                }
            }

            let y = solve_upper_triangular(&workspace.hessenberg, &workspace.gamma, vectors_used)?;
            for (i, &coefficient) in y.iter().enumerate() {
                axpy(x0, coefficient, &workspace.basis[i]);
            }

            let ax = matvec(x0)?;
            assign_difference(&mut workspace.residual, b, &ax);
            rho = norm(&workspace.residual);
            check_finite(rho, "residual")?;

            // Convergence is decided on the recomputed residual, never on the
            // least-squares estimate alone: the estimate can reach zero on a
            // degenerate cycle while `b − A·x` is untouched, and reporting
            // success from it would return an unsolved system.
            if check_convergence(&self.config, rho, b_norm) {
                return Ok(converged_info(self.iteration_count, rho, b_norm));
            }
            self.residual_history.push(rho);

            if breakdown {
                // The subspace is invariant under A, so no further Krylov
                // vector exists and restarting would rebuild the same cycle.
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    format!(
                        "GMRES Krylov subspace exhausted at residual {rho:.3e}, above tolerance"
                    ),
                )));
            }
        }

        Err(KwaversError::Numerical(NumericalError::InvalidOperation(
            format!(
                "GMRES did not converge in {} iterations. Final residual: {:.3e}",
                self.iteration_count, rho
            ),
        )))
    }

    /// Get residual history.
    ///
    /// Entries recorded inside a restart cycle are the Arnoldi least-squares
    /// estimates; each restart boundary additionally records the recomputed
    /// residual `‖b − A·x‖`. Without preconditioning the two agree to rounding.
    #[must_use]
    pub fn residual_history(&self) -> &[f64] {
        &self.residual_history
    }

    /// Get total iteration count.
    #[must_use]
    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }

    /// Compute Givens rotation (c, s) such that [-s c][a b]ᵀ = [r 0]ᵀ.
    ///
    /// The tangent form divides the smaller magnitude by the larger, so the
    /// intermediate `√(1 + t²)` neither overflows nor underflows for any finite
    /// input.
    pub(super) fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
        if b == 0.0 {
            (1.0, 0.0)
        } else if a.abs() < b.abs() {
            let tangent = a / b;
            let s = 1.0 / tangent.mul_add(tangent, 1.0).sqrt();
            (tangent * s, s)
        } else {
            let tangent = b / a;
            let c = 1.0 / tangent.mul_add(tangent, 1.0).sqrt();
            (c, tangent * c)
        }
    }
}

fn converged_info(iterations: usize, residual: f64, b_norm: f64) -> GmresConvergenceInfo {
    GmresConvergenceInfo {
        converged: true,
        iterations,
        final_residual: residual,
        relative_residual: relative_residual(residual, b_norm),
    }
}

/// Relative residual, defined as `‖r‖` itself when `b` vanishes so that a
/// zero right-hand side does not divide by zero.
fn relative_residual(residual: f64, b_norm: f64) -> f64 {
    if b_norm > 0.0 {
        residual / b_norm
    } else {
        residual
    }
}

fn check_convergence(config: &GMRESConfig, residual: f64, b_norm: f64) -> bool {
    residual < config.absolute_tolerance
        || relative_residual(residual, b_norm) < config.relative_tolerance
}

fn check_finite(value: f64, what: &str) -> KwaversResult<()> {
    if value.is_finite() {
        return Ok(());
    }
    Err(KwaversError::Numerical(NumericalError::InvalidOperation(
        format!("GMRES produced a non-finite {what}"),
    )))
}

fn norm(a: &Array3<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `target ← target + scale · source`, without allocating.
fn axpy(target: &mut Array3<f64>, scale: f64, source: &Array3<f64>) {
    for (value, &increment) in target.iter_mut().zip(source.iter()) {
        *value = scale.mul_add(increment, *value);
    }
}

/// `target ← scale · source`, without allocating.
fn scale_into(target: &mut Array3<f64>, source: &Array3<f64>, scale: f64) {
    for (value, &source_value) in target.iter_mut().zip(source.iter()) {
        *value = source_value * scale;
    }
}

/// `target ← lhs − rhs`, without allocating.
fn assign_difference(target: &mut Array3<f64>, lhs: &Array3<f64>, rhs: &Array3<f64>) {
    for ((value, &left), &right) in target.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        *value = left - right;
    }
}

/// Solve the leading `k × k` triangular block `R·y = g`.
fn solve_upper_triangular(h: &[Vec<f64>], g: &[f64], k: usize) -> KwaversResult<Vec<f64>> {
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = g[i];
        for j in i + 1..k {
            sum -= h[i][j] * y[j];
        }
        let diagonal = h[i][i];
        if diagonal == 0.0 || !diagonal.is_finite() || !sum.is_finite() {
            return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                format!("GMRES: singular or non-finite Hessenberg factor at row {i}"),
            )));
        }
        y[i] = sum / diagonal;
        check_finite(y[i], "least-squares coefficient")?;
    }
    Ok(y)
}
