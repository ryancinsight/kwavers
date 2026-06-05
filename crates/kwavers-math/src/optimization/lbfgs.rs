//! Limited-memory BFGS (L-BFGS) quasi-Newton optimiser.
//!
//! L-BFGS approximates the inverse Hessian from the last `m` gradient/step pairs
//! `(s_k, y_k)` and computes a descent direction by the Nocedal two-loop
//! recursion, giving super-linear convergence without storing or inverting a
//! dense Hessian. It is the standard refinement step for full-waveform inversion
//! and PINN training (Inverse Problems §9.1).
//!
//! # References
//! - Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
//!   *Math. Comp.*, 35(151), 773–782.
//! - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Alg. 7.4–7.5.

/// L-BFGS configuration.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsConfig {
    /// Number of `(s, y)` correction pairs kept (`m`).
    pub memory: usize,
    /// Maximum outer iterations.
    pub max_iters: usize,
    /// Convergence tolerance on the gradient infinity-norm.
    pub gtol: f64,
    /// Armijo sufficient-decrease constant `c₁ ∈ (0, 1)`.
    pub c1: f64,
    /// Maximum backtracking line-search steps per iteration.
    pub max_line_search: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            memory: 8,
            max_iters: 200,
            gtol: 1e-8,
            c1: 1e-4,
            max_line_search: 30,
        }
    }
}

/// Result of an L-BFGS run.
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Minimiser estimate.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fx: f64,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Whether the gradient tolerance was met.
    pub converged: bool,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Minimise `f` with gradient `grad`, starting from `x0`, via L-BFGS.
///
/// `f: &[f64] -> f64` is the objective; `grad: &[f64] -> Vec<f64>` its gradient.
/// Returns the minimiser, the objective there, the iteration count, and whether
/// the gradient infinity-norm fell below `config.gtol`.
pub fn minimize<F, G>(x0: Vec<f64>, mut f: F, mut grad: G, config: LbfgsConfig) -> LbfgsResult
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0;
    let mut fx = f(&x);
    let mut g = grad(&x);

    // history of correction pairs (most-recent last)
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(config.memory);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(config.memory);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(config.memory);

    if inf_norm(&g) < config.gtol {
        return LbfgsResult {
            x,
            fx,
            iterations: 0,
            converged: true,
        };
    }

    for it in 1..=config.max_iters {
        // ---- two-loop recursion: direction d = -H·g ----
        let mut q = g.clone();
        let k = s_hist.len();
        let mut alpha = vec![0.0_f64; k];
        for i in (0..k).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            for j in 0..n {
                q[j] -= a * y_hist[i][j];
            }
        }
        // initial Hessian scaling γ = (sᵀy)/(yᵀy) from the newest pair
        let gamma = if k > 0 {
            let last = k - 1;
            let sy = dot(&s_hist[last], &y_hist[last]);
            let yy = dot(&y_hist[last], &y_hist[last]);
            if yy > 0.0 {
                sy / yy
            } else {
                1.0
            }
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();
        for i in 0..k {
            let beta = rho_hist[i] * dot(&y_hist[i], &r);
            let coef = alpha[i] - beta;
            for j in 0..n {
                r[j] += coef * s_hist[i][j];
            }
        }
        let dir: Vec<f64> = r.iter().map(|&ri| -ri).collect();

        // ---- Armijo backtracking line search ----
        let gd = dot(&g, &dir); // directional derivative (< 0 for a descent dir)
        let mut step = if k == 0 {
            // first iteration: scale the steepest-descent step
            (1.0 / inf_norm(&g)).min(1.0)
        } else {
            1.0
        };
        let mut x_new = x.clone();
        let mut fx_new = fx;
        let mut accepted = false;
        for _ in 0..config.max_line_search {
            for j in 0..n {
                x_new[j] = x[j] + step * dir[j];
            }
            fx_new = f(&x_new);
            if fx_new <= fx + config.c1 * step * gd {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            // line search failed to make progress → stop
            return LbfgsResult {
                x,
                fx,
                iterations: it,
                converged: inf_norm(&g) < config.gtol,
            };
        }

        let g_new = grad(&x_new);

        // ---- store correction pair if the curvature condition holds ----
        let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
        let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if s_hist.len() == config.memory {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s);
            y_hist.push(y);
        }

        x = x_new;
        fx = fx_new;
        g = g_new;

        if inf_norm(&g) < config.gtol {
            return LbfgsResult {
                x,
                fx,
                iterations: it,
                converged: true,
            };
        }
    }

    LbfgsResult {
        x,
        fx,
        iterations: config.max_iters,
        converged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x) = ½ xᵀA x − bᵀx with SPD A → minimiser x* = A⁻¹b.
    #[test]
    fn lbfgs_minimises_spd_quadratic() {
        // A = [[3,1],[1,2]], b = [1,1]; x* = [0.2, 0.4]
        let a = [[3.0, 1.0], [1.0, 2.0]];
        let b = [1.0, 1.0];
        let f = |x: &[f64]| {
            let ax0 = a[0][0] * x[0] + a[0][1] * x[1];
            let ax1 = a[1][0] * x[0] + a[1][1] * x[1];
            0.5 * (x[0] * ax0 + x[1] * ax1) - (b[0] * x[0] + b[1] * x[1])
        };
        let grad = |x: &[f64]| {
            vec![
                a[0][0] * x[0] + a[0][1] * x[1] - b[0],
                a[1][0] * x[0] + a[1][1] * x[1] - b[1],
            ]
        };
        let res = minimize(vec![0.0, 0.0], f, grad, LbfgsConfig::default());
        assert!(res.converged, "L-BFGS should converge on a quadratic");
        assert!((res.x[0] - 0.2).abs() < 1e-6, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 0.4).abs() < 1e-6, "x1 = {}", res.x[1]);
        // a quadratic is solved in very few quasi-Newton steps
        assert!(res.iterations <= 15, "took {} iters", res.iterations);
    }

    /// Separable convex objective Σ (xᵢ − tᵢ)⁴ → minimiser is t.
    #[test]
    fn lbfgs_minimises_quartic_well() {
        let t = [1.5, -2.0, 0.7, 3.1];
        let f = |x: &[f64]| {
            x.iter()
                .zip(t)
                .map(|(xi, ti)| (xi - ti).powi(4))
                .sum::<f64>()
        };
        let grad = |x: &[f64]| {
            x.iter()
                .zip(t)
                .map(|(xi, ti)| 4.0 * (xi - ti).powi(3))
                .collect::<Vec<_>>()
        };
        let cfg = LbfgsConfig {
            gtol: 1e-10,
            max_iters: 500,
            ..Default::default()
        };
        let res = minimize(vec![0.0; 4], f, grad, cfg);
        for (xi, ti) in res.x.iter().zip(t) {
            assert!((xi - ti).abs() < 1e-2, "got {xi}, want {ti}");
        }
    }

    #[test]
    fn lbfgs_returns_immediately_at_optimum() {
        // start at the minimiser of ½‖x‖² (gradient zero)
        let f = |x: &[f64]| 0.5 * x.iter().map(|v| v * v).sum::<f64>();
        let grad = |x: &[f64]| x.to_vec();
        let res = minimize(vec![0.0, 0.0, 0.0], f, grad, LbfgsConfig::default());
        assert!(res.converged);
        assert_eq!(res.iterations, 0);
    }
}
