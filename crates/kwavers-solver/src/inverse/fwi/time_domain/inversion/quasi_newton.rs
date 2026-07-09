//! Quasi-Newton (L-BFGS) full-waveform-inversion driver.
//!
//! Replaces the steepest-descent search direction of [`FwiProcessor::invert`]
//! with the limited-memory BFGS direction `d = −H·g`, where the implicit
//! inverse-Hessian `H` is reconstructed from the last `m` model/gradient
//! correction pairs by the Nocedal two-loop recursion. The recursion itself is
//! the canonical [`LbfgsMemory`] from `kwavers_math` (SSOT); this module owns
//! only the FWI-specific glue: flattening the 3-D model, running the
//! forward+adjoint gradient pass, and an Armijo projected line search.
//!
//! L-BFGS gives super-linear convergence near the minimiser and conditions the
//! gradient by the (approximate) inverse Hessian, which mitigates the geometric
//! spreading / illumination imbalance that slows raw gradient descent in FWI
//! (Métivier & Brossier 2016; Nocedal & Wright 2006 §7.2).
//!
//! # References
//! - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Alg. 7.4–7.5.
//! - Métivier, L., & Brossier, R. (2016). "The SEISCOPE optimization toolbox."
//!   *Geophysics*, 81(2), F1–F15.

use super::super::{geometry::FwiGeometry, FwiProcessor};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_math::optimization::LbfgsMemory;
use leto::Array3;

/// Maximum Armijo backtracking halvings per outer L-BFGS iteration.
const MAX_LINE_SEARCH: usize = 12;
/// Armijo sufficient-decrease constant `c₁ ∈ (0, 1)`.
const ARMIJO_C1: f64 = 1e-4;

impl FwiProcessor {
    /// Perform Full Waveform Inversion with the L-BFGS quasi-Newton update.
    ///
    /// Single-source variant of [`Self::invert`] that uses the limited-memory
    /// BFGS search direction instead of normalized steepest descent. Keeps
    /// `memory` correction pairs (8–20 is typical; larger `m` improves the
    /// Hessian approximation at higher memory cost). The misfit functional,
    /// regularization, model constraints, and convergence/tolerance contract are
    /// identical to [`Self::invert`].
    ///
    /// The gradient used to drive L-BFGS is the **un-normalized** smoothed,
    /// regularized reduced gradient `g = +∂J/∂c` from
    /// [`Self::misfit_and_gradient`]: the curvature pairs `(s, y = Δg)` must
    /// retain physical scaling for the inverse-Hessian estimate `γ = sᵀy/yᵀy`
    /// to be meaningful. The first iteration (empty memory ⇒ steepest descent
    /// direction `−g`) is scaled by `parameters.step_size / ‖g‖∞` so its model
    /// change matches the steepest-descent driver; subsequent iterations try a
    /// unit step first, as the two-loop recursion already carries the units.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the geometry is invalid or
    ///   `nt < 3`.
    /// - Propagates any [`KwaversError`] from the forward/adjoint solve, the
    ///   misfit evaluation, or regularization.
    pub fn invert_lbfgs(
        &self,
        observed_data: &leto::Array2<f64>,
        initial_model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
        memory: usize,
    ) -> KwaversResult<Array3<f64>> {
        geometry.validate(grid, self.parameters.nt)?;
        if self.parameters.nt < 3 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires at least 3 time samples to form a second derivative"
                        .to_owned(),
                },
            ));
        }

        let dim = initial_model.shape();
        let mut model = initial_model.clone();
        self.apply_model_constraints(&mut model);

        let mut mem = LbfgsMemory::new(memory);

        // Initial objective and gradient at the (constrained) starting model.
        // L-BFGS uses a constant prior weight (dtv_scale = 1.0): an iteration-
        // varying regularizer would corrupt the secant curvature pairs (s, y=Δg).
        let (mut objective, grad_arr) =
            self.misfit_and_gradient(&model, observed_data, geometry, grid, 1.0)?;
        let mut x: Vec<f64> = model.iter().copied().collect();
        let mut g: Vec<f64> = grad_arr.iter().copied().collect();

        // Convergence is judged RELATIVE to the initial gradient norm: the FWI
        // gradient magnitude is problem-scale-dependent (e.g. the self-adjoint
        // engine's dt²·ρc² source injection yields ‖g‖∞ ~ 1e-18 for unit-amplitude
        // data), so an absolute floor would falsely report "zero gradient" and
        // stall immediately. A finite, strictly-positive `g0_inf` is guaranteed by
        // the non-stationary-start check below.
        let g0_inf = inf_norm(&g);
        const GRAD_REL_TOL: f64 = 1e-8;
        if g0_inf <= f64::MIN_POSITIVE {
            log::info!("FWI(L-BFGS): initial gradient is identically zero, stopping");
            return Array3::from_shape_vec(dim, x).map_err(|e| {
                KwaversError::Validation(ValidationError::ConstraintViolation {
                    message: format!("L-BFGS model reshape failed: {e}"),
                })
            });
        }

        for iteration in 0..self.parameters.max_iterations {
            let g_inf = inf_norm(&g);
            if g_inf <= GRAD_REL_TOL * g0_inf {
                log::info!(
                    "FWI(L-BFGS) iter {iteration}: gradient reduced below {GRAD_REL_TOL:e}·‖g₀‖∞, stopping"
                );
                break;
            }

            // Search direction d = −H·g (steepest descent on the first step).
            let dir = mem.direction(&g);
            let gd = dot(&g, &dir); // directional derivative; < 0 for a descent dir
            if gd >= 0.0 {
                log::info!(
                    "FWI(L-BFGS) iter {iteration}: non-descent direction (gᵀd={gd:.3e}); stopping"
                );
                break;
            }

            // First iteration: scale the steepest step to the configured step
            // size; afterwards the L-BFGS scaling carries the units → try α = 1.
            let mut step = if mem.is_empty() {
                (self.parameters.step_size / g_inf).max(f64::MIN_POSITIVE)
            } else {
                1.0
            };

            let mut accepted: Option<Vec<f64>> = None;
            for _ in 0..MAX_LINE_SEARCH {
                let mut trial = Array3::from_shape_vec(
                    dim,
                    x.iter()
                        .zip(&dir)
                        .map(|(&xi, &di)| step.mul_add(di, xi))
                        .collect(),
                )
                .expect("trial model shares the model shape");
                self.apply_model_constraints(&mut trial);
                let trial_obj = self.compute_objective(&trial, observed_data, geometry, grid)?;
                if trial_obj <= objective + ARMIJO_C1 * step * gd {
                    accepted = Some(trial.iter().copied().collect());
                    break;
                }
                step *= 0.5;
            }

            let Some(x_new) = accepted else {
                log::info!(
                    "FWI(L-BFGS) stalled at iter {iteration}: line search found no descent \
                     (J={objective:.6e})"
                );
                break;
            };

            // Recompute the gradient at the accepted model to form the curvature
            // pair and to drive the next iteration.
            let model_new = Array3::from_shape_vec(dim, x_new.clone())
                .expect("accepted model shares the model shape");
            let (obj_grad, grad_new_arr) =
                self.misfit_and_gradient(&model_new, observed_data, geometry, grid, 1.0)?;
            let g_new: Vec<f64> = grad_new_arr.iter().copied().collect();

            // s = xₖ₊₁ − xₖ,  y = ∇Jₖ₊₁ − ∇Jₖ.
            let s: Vec<f64> = x_new.iter().zip(&x).map(|(&a, &b)| a - b).collect();
            let y: Vec<f64> = g_new.iter().zip(&g).map(|(&a, &b)| a - b).collect();
            mem.push(s, y);

            let relative_change = (objective - obj_grad).abs() / objective.max(f64::EPSILON);
            log::info!(
                "FWI(L-BFGS) iter {iteration}: J={objective:.6e} step={step:.6e} \
                 ‖g‖∞={g_inf:.3e}"
            );

            x = x_new;
            g = g_new;
            objective = obj_grad;

            if relative_change < self.parameters.tolerance {
                log::info!(
                    "FWI(L-BFGS) converged after {iteration} iterations with objective: \
                     {objective:.6e}"
                );
                break;
            }
        }

        Array3::from_shape_vec(dim, x).map_err(|e| {
            KwaversError::Validation(ValidationError::ConstraintViolation {
                message: format!("L-BFGS model reshape failed: {e}"),
            })
        })
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}
