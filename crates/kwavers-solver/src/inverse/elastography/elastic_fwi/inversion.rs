//! Regularization, combined objective/gradient, and the steepest-descent
//! inversion loop with Armijo line search (ADR 033 increment 3).

use kwavers_core::error::KwaversResult;
use kwavers_math::optimization::LbfgsMemory;
use leto::Array3;

use super::ElasticFwi;

/// Huber smoothing for the isotropic-TV denominator (`ε²`), in Pa².
const TV_EPS_SQ: f64 = 1.0e-6;

impl ElasticFwi {
    /// Total objective `J_data(μ) + R(μ)` (data misfit plus regularization).
    ///
    /// # Errors
    /// Propagates solver errors.
    pub fn objective(&mut self, mu: &Array3<f64>) -> KwaversResult<f64> {
        Ok(self.forward_misfit(mu)? + self.regularization_penalty(mu))
    }

    /// Total objective and its gradient `∂J/∂μ = K_μ + ∂R/∂μ`.
    ///
    /// # Errors
    /// Propagates solver errors.
    pub fn misfit_and_gradient(&mut self, mu: &Array3<f64>) -> KwaversResult<(f64, Array3<f64>)> {
        let (j_data, mut grad) = if self.config.precond_eps > 0.0 {
            self.data_misfit_and_preconditioned_gradient(mu)?
        } else {
            self.data_misfit_and_gradient(mu)?
        };
        let penalty = self.regularization_penalty(mu);
        self.add_regularization_gradient(&mut grad, mu);
        Ok((j_data + penalty, grad))
    }

    /// Reconstruct `μ` by steepest descent with Armijo backtracking, starting
    /// from the engine's initial model. Returns the final `μ` map and leaves it
    /// installed on the internal solver.
    ///
    /// Each iteration: compute the regularized gradient, normalize it by its
    /// max norm (step-size-invariant), and accept the largest backtracked step
    /// that strictly decreases the total objective; stop when no step improves.
    ///
    /// # Errors
    /// Propagates solver errors.
    pub fn run(&mut self) -> KwaversResult<Array3<f64>> {
        let mut mu = self.solver.mu().clone();
        let mut objective = self.objective(&mu)?;

        for _iter in 0..self.config.iterations {
            let (_j, grad) = self.misfit_and_gradient(&mu)?;
            let gmax = grad.iter().fold(0.0_f64, |m, &g| m.max(g.abs()));
            if gmax <= 0.0 {
                break;
            }

            let mut step = self.config.step_size;
            let mut improved = false;
            for _ls in 0..12 {
                let alpha = step / gmax;
                let trial = self.stepped_model(&mu, &grad, alpha);
                let j_trial = self.objective(&trial)?;
                if j_trial < objective {
                    mu = trial;
                    objective = j_trial;
                    improved = true;
                    break;
                }
                step *= 0.5;
            }
            if !improved {
                break;
            }
        }

        self.solver.set_mu(&mu)?;
        Ok(mu)
    }

    /// Reconstruct `μ` by L-BFGS quasi-Newton with Armijo backtracking — the
    /// faster-converging alternative to [`Self::run`] (steepest descent), and the
    /// ADR-033 increment-5 optimizer. Uses the **true** gradient `∂J/∂μ` (raw
    /// `K_μ` + regularization, *no* illumination preconditioner: the inverse-
    /// Hessian approximation subsumes that role), so the Armijo directional
    /// derivative `gᵀd` is exact. `memory` is the number of stored curvature pairs
    /// (typically 5–10). Mirrors the acoustic `FwiProcessor::invert_lbfgs`.
    ///
    /// # Errors
    /// Propagates solver errors.
    pub fn run_lbfgs(&mut self, memory: usize) -> KwaversResult<Array3<f64>> {
        let dim = self.solver.mu().shape();
        let mut model = self.solver.mu().clone();
        for m in model.iter_mut() {
            *m = m.clamp(self.config.mu_min, self.config.mu_max);
        }

        let mut mem = LbfgsMemory::new(memory);
        let (mut objective, grad0) = self.misfit_and_true_gradient(&model)?;
        let mut x: Vec<f64> = model.iter().copied().collect();
        let mut g: Vec<f64> = grad0.iter().copied().collect();

        let g0_inf = inf_norm(&g);
        if g0_inf <= f64::MIN_POSITIVE {
            return Ok(model);
        }
        const GRAD_REL_TOL: f64 = 1.0e-8;

        for _iter in 0..self.config.iterations {
            if inf_norm(&g) <= GRAD_REL_TOL * g0_inf {
                break;
            }
            let dir = mem.direction(&g); // descent direction d = −H·g
            let gd = dot(&g, &dir);
            if gd >= 0.0 {
                break; // not a descent direction
            }
            // First step scales steepest descent to the configured step; once
            // L-BFGS has curvature it carries the units, so start at α = 1.
            let mut step = if mem.is_empty() {
                (self.config.step_size / inf_norm(&g)).max(f64::MIN_POSITIVE)
            } else {
                1.0
            };
            let mut accepted: Option<Vec<f64>> = None;
            for _ in 0..12 {
                let trial: Vec<f64> = x
                    .iter()
                    .zip(&dir)
                    .map(|(&xi, &di)| {
                        step.mul_add(di, xi)
                            .clamp(self.config.mu_min, self.config.mu_max)
                    })
                    .collect();
                let trial_arr =
                    Array3::from_shape_vec(dim, trial.clone()).expect("trial shares model shape");
                if self.objective(&trial_arr)? <= 1e-4f64.mul_add(step * gd, objective) {
                    accepted = Some(trial);
                    break;
                }
                step *= 0.5;
            }
            let Some(x_new) = accepted else {
                break; // line search stalled
            };
            let model_new = Array3::from_shape_vec(dim, x_new.clone()).expect("model shares shape");
            let (obj_new, grad_new) = self.misfit_and_true_gradient(&model_new)?;
            let g_new: Vec<f64> = grad_new.iter().copied().collect();
            // Curvature pair: s = xₖ₊₁ − xₖ, y = ∇Jₖ₊₁ − ∇Jₖ.
            let s: Vec<f64> = x_new.iter().zip(&x).map(|(&a, &b)| a - b).collect();
            let y: Vec<f64> = g_new.iter().zip(&g).map(|(&a, &b)| a - b).collect();
            mem.push(s, y);
            x = x_new;
            g = g_new;
            objective = obj_new;
        }

        let mu = Array3::from_shape_vec(dim, x).expect("model shares shape");
        self.solver.set_mu(&mu)?;
        Ok(mu)
    }

    /// Total objective and the **true** gradient `∂J/∂μ` (raw `K_μ` +
    /// regularization, no illumination preconditioner) — the consistent
    /// objective/gradient pair L-BFGS requires for a valid Armijo line search.
    ///
    /// # Errors
    /// Propagates solver errors.
    fn misfit_and_true_gradient(&mut self, mu: &Array3<f64>) -> KwaversResult<(f64, Array3<f64>)> {
        let (j_data, mut grad) = self.data_misfit_and_gradient(mu)?;
        let penalty = self.regularization_penalty(mu);
        self.add_regularization_gradient(&mut grad, mu);
        Ok((j_data + penalty, grad))
    }

    /// `clamp(μ − α·grad, μ_min, μ_max)`.
    fn stepped_model(&self, mu: &Array3<f64>, grad: &Array3<f64>, alpha: f64) -> Array3<f64> {
        let mut out = mu.clone();
        for (m, g) in out.iter_mut().zip(grad.iter()) {
            *m = (*m - alpha * g).clamp(self.config.mu_min, self.config.mu_max);
        }
        out
    }

    /// Regularization penalty `R(μ) = (λ_Tik/2)‖μ − μ₀‖² + λ_TV·TV(μ)` (isotropic
    /// Huber TV over the in-plane slice).
    fn regularization_penalty(&self, mu: &Array3<f64>) -> f64 {
        let mut penalty = 0.0;
        if self.config.tikhonov_weight > 0.0 {
            let mut acc = 0.0;
            for (m, m0) in mu.iter().zip(self.mu_start.iter()) {
                let d = m - m0;
                acc += d * d;
            }
            penalty += 0.5 * self.config.tikhonov_weight * acc;
        }
        if self.config.tv_weight > 0.0 {
            penalty += self.config.tv_weight * tv_functional(mu);
        }
        penalty
    }

    /// Accumulate `∂R/∂μ` into `grad`.
    ///
    /// Tikhonov: `λ_Tik·(μ − μ₀)`. TV: the ROF divergence (see [`add_tv_gradient`]).
    fn add_regularization_gradient(&self, grad: &mut Array3<f64>, mu: &Array3<f64>) {
        if self.config.tikhonov_weight > 0.0 {
            let w = self.config.tikhonov_weight;
            for ((g, &m), &m0) in grad.iter_mut().zip(mu.iter()).zip(self.mu_start.iter()) {
                *g += w * (m - m0);
            }
        }
        if self.config.tv_weight > 0.0 {
            add_tv_gradient(grad, mu, self.config.tv_weight);
        }
    }
}

/// Max-norm `‖v‖∞`.
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Euclidean dot product `a·b`.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Isotropic Huber-smoothed total variation `Σ √(ε² + (∂_x μ)² + (∂_y μ)²)` over
/// the `k = 0` plane (forward differences).
fn tv_functional(mu: &Array3<f64>) -> f64 {
    let [nx, ny, _nz] = mu.shape();
    let mut acc = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            let dx = if i + 1 < nx {
                mu[[i + 1, j, 0]] - mu[[i, j, 0]]
            } else {
                0.0
            };
            let dy = if j + 1 < ny {
                mu[[i, j + 1, 0]] - mu[[i, j, 0]]
            } else {
                0.0
            };
            acc += dx.mul_add(dx, dy.mul_add(dy, TV_EPS_SQ)).sqrt();
        }
    }
    acc
}

/// Add `λ_TV · ∂TV/∂μ` (standard Rudin–Osher–Fatemi divergence) into `grad`.
fn add_tv_gradient(grad: &mut Array3<f64>, mu: &Array3<f64>, weight: f64) {
    let [nx, ny, _nz] = mu.shape();
    let fwd_dx = |i: usize, j: usize| {
        if i + 1 < nx {
            mu[[i + 1, j, 0]] - mu[[i, j, 0]]
        } else {
            0.0
        }
    };
    let fwd_dy = |i: usize, j: usize| {
        if j + 1 < ny {
            mu[[i, j + 1, 0]] - mu[[i, j, 0]]
        } else {
            0.0
        }
    };
    let denom = |i: usize, j: usize| {
        let dx = fwd_dx(i, j);
        let dy = fwd_dy(i, j);
        dx.mul_add(dx, dy.mul_add(dy, TV_EPS_SQ)).sqrt()
    };

    for i in 0..nx {
        for j in 0..ny {
            let d0 = denom(i, j);
            // Contribution of pixel (i,j)'s own forward differences.
            let mut g = -(fwd_dx(i, j) + fwd_dy(i, j)) / d0;
            // Contribution through the left neighbour's x-difference.
            if i > 0 {
                g += (mu[[i, j, 0]] - mu[[i - 1, j, 0]]) / denom(i - 1, j);
            }
            // Contribution through the lower neighbour's y-difference.
            if j > 0 {
                g += (mu[[i, j, 0]] - mu[[i, j - 1, 0]]) / denom(i, j - 1);
            }
            grad[[i, j, 0]] += weight * g;
        }
    }
}
