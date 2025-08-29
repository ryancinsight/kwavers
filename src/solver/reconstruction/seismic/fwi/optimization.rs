//! Optimization algorithms for FWI
//! Based on Nocedal & Wright (2006): "Numerical Optimization"

use ndarray::{Array3, Zip};

/// Line search methods for step size selection
pub struct LineSearch {
    /// Armijo constant for sufficient decrease
    c1: f64,
    /// Wolfe constant for curvature condition
    c2: f64,
    /// Maximum iterations
    max_iterations: usize,
}

impl LineSearch {
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
    pub fn backtracking(
        &self,
        direction: &Array3<f64>,
        gradient: &Array3<f64>,
        initial_step: f64,
    ) -> f64 {
        let mut alpha = initial_step;
        let shrink_factor = 0.5;

        // TODO: Implement proper backtracking
        // while !armijo_condition(alpha) {
        //     alpha *= shrink_factor;
        // }

        alpha * shrink_factor
    }
}

/// Conjugate gradient optimizer
pub struct ConjugateGradient {
    /// Previous gradient for beta computation
    previous_gradient: Option<Array3<f64>>,
    /// Previous search direction
    previous_direction: Option<Array3<f64>>,
}

impl ConjugateGradient {
    pub fn new() -> Self {
        Self {
            previous_gradient: None,
            previous_direction: None,
        }
    }

    /// Compute search direction using Polak-Ribière formula
    pub fn compute_direction(&mut self, gradient: &Array3<f64>) -> Array3<f64> {
        let direction = if let (Some(ref prev_grad), Some(ref prev_dir)) =
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
pub struct Lbfgs {
    /// Number of stored vector pairs
    memory_size: usize,
    /// Stored gradient differences
    s_vectors: Vec<Array3<f64>>,
    /// Stored gradient changes
    y_vectors: Vec<Array3<f64>>,
}

impl Lbfgs {
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
        // TODO: Implement L-BFGS two-loop recursion
        // This is memory-efficient for large problems

        -gradient.clone()
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
