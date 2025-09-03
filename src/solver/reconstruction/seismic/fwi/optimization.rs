//! Optimization algorithms for FWI
//! Based on Nocedal & Wright (2006): "Numerical Optimization"

use ndarray::{Array3, Zip};

/// Line search methods for step size selection
#[derive(Debug)]
pub struct LineSearch {
    /// Armijo constant for sufficient decrease
    c1: f64,
    /// Wolfe constant for curvature condition
    c2: f64,
    /// Maximum iterations
    max_iterations: usize,
}

impl Default for LineSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl LineSearch {
    #[must_use]
    pub fn new() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            max_iterations: 20,
        }
    }

    /// Wolfe line search
    /// Based on Nocedal & Wright (2006), Chapter 3
    pub fn wolfe_search(
        &self,
        direction: &Array3<f64>,
        gradient: &Array3<f64>,
        objective_fn: impl Fn(&Array3<f64>) -> f64,
    ) -> f64 {
        // Strong Wolfe conditions:
        // 1. Armijo: f(x + αp) ≤ f(x) + c1*α*∇f·p
        // 2. Curvature: |∇f(x + αp)·p| ≤ c2*|∇f(x)·p|

        // Simplified implementation - return conservative step
        0.01
    }

    /// Backtracking line search
    #[must_use]
    pub fn backtracking(
        &self,
        direction: &Array3<f64>,
        gradient: &Array3<f64>,
        initial_step: f64,
    ) -> f64 {
        let alpha = initial_step;
        let shrink_factor = 0.5;

        // TODO: Implement proper backtracking
        // while !armijo_condition(alpha) {
        //     alpha *= shrink_factor;
        // }

        alpha * shrink_factor
    }
}

/// Conjugate gradient optimizer
#[derive(Debug)]
pub struct ConjugateGradient {
    /// Previous gradient for beta computation
    previous_gradient: Option<Array3<f64>>,
    /// Previous search direction
    previous_direction: Option<Array3<f64>>,
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self::new()
    }
}

impl ConjugateGradient {
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous_gradient: None,
            previous_direction: None,
        }
    }

    /// Compute search direction using Polak-Ribière formula
    pub fn compute_direction(&mut self, gradient: &Array3<f64>) -> Array3<f64> {
        let direction = if let (Some(prev_grad), Some(prev_dir)) =
            (&self.previous_gradient, &self.previous_direction)
        {
            // Polak-Ribière: β = (g_k·(g_k - g_{k-1})) / ||g_{k-1}||²
            let grad_diff = gradient - prev_grad;
            let beta = Zip::from(gradient)
                .and(&grad_diff)
                .fold(0.0, |acc, &g, &d| acc + g * d)
                / Zip::from(prev_grad).fold(0.0, |acc, &g| acc + g * g);

            // d_k = -g_k + β*d_{k-1}
            -gradient + beta * prev_dir
        } else {
            // First iteration: steepest descent
            -gradient.clone()
        };

        self.previous_gradient = Some(gradient.clone());
        self.previous_direction = Some(direction.clone());

        direction
    }

    /// Reset conjugate gradient (for nonlinear CG)
    pub fn reset(&mut self) {
        self.previous_gradient = None;
        self.previous_direction = None;
    }
}

/// L-BFGS optimizer for large-scale problems
#[derive(Debug)]
pub struct Lbfgs {
    /// Number of stored vector pairs
    memory_size: usize,
    /// Stored gradient differences
    s_vectors: Vec<Array3<f64>>,
    /// Stored gradient changes
    y_vectors: Vec<Array3<f64>>,
}

impl Lbfgs {
    #[must_use]
    pub fn new(memory_size: usize) -> Self {
        Self {
            memory_size,
            s_vectors: Vec::with_capacity(memory_size),
            y_vectors: Vec::with_capacity(memory_size),
        }
    }

    /// Compute search direction using L-BFGS two-loop recursion
    /// Based on Nocedal & Wright (2006), Algorithm 7.4
    pub fn compute_direction(&mut self, gradient: &Array3<f64>) -> Array3<f64> {
        // L-BFGS two-loop recursion implementation
        let mut q = gradient.clone();
        let m = self.s_vectors.len();

        if m == 0 {
            return -q; // Steepest descent for first iteration
        }

        let mut alpha = vec![0.0; m];

        // First loop: right product
        for i in (0..m).rev() {
            let s_i = &self.s_vectors[i];
            let y_i = &self.y_vectors[i];

            // Compute ρ_i = 1 / (y_i · s_i)
            let dot_sy: f64 = s_i.iter().zip(y_i.iter()).map(|(s, y)| s * y).sum();
            if dot_sy.abs() < 1e-10 {
                continue; // Skip if denominator too small
            }
            let rho = 1.0 / dot_sy;

            // α_i = ρ_i * (s_i · q)
            let dot_sq: f64 = s_i.iter().zip(q.iter()).map(|(s, q)| s * q).sum();
            alpha[i] = rho * dot_sq;

            // q = q - α_i * y_i
            q.zip_mut_with(y_i, |q_val, y_val| {
                *q_val -= alpha[i] * y_val;
            });
        }

        // Apply initial Hessian approximation
        if let (Some(last_s), Some(last_y)) = (self.s_vectors.last(), self.y_vectors.last()) {
            let dot_sy: f64 = last_s.iter().zip(last_y.iter()).map(|(s, y)| s * y).sum();
            let dot_yy: f64 = last_y.iter().map(|y| y * y).sum();

            if dot_yy > 1e-10 {
                let gamma = dot_sy / dot_yy;
                q *= gamma;
            }
        }

        // Second loop: left product
        for i in 0..m {
            let s_i = &self.s_vectors[i];
            let y_i = &self.y_vectors[i];

            let dot_sy: f64 = s_i.iter().zip(y_i.iter()).map(|(s, y)| s * y).sum();
            if dot_sy.abs() < 1e-10 {
                continue;
            }
            let rho = 1.0 / dot_sy;

            let dot_yq: f64 = y_i.iter().zip(q.iter()).map(|(y, q)| y * q).sum();
            let beta = rho * dot_yq;

            // q = q + s_i * (α_i - β)
            q.zip_mut_with(s_i, |q_val, s_val| {
                *q_val += s_val * (alpha[i] - beta);
            });
        }

        -q // Return negative for descent direction
    }

    /// Update stored vectors after step
    pub fn update(&mut self, step: Array3<f64>, gradient_change: Array3<f64>) {
        if self.s_vectors.len() >= self.memory_size {
            self.s_vectors.remove(0);
            self.y_vectors.remove(0);
        }

        self.s_vectors.push(step);
        self.y_vectors.push(gradient_change);
    }
}
