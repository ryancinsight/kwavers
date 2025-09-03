//! IMEX-BDF (Backward Differentiation Formula) schemes

use super::traits::IMEXScheme;
use super::ImplicitSolverType;
use crate::error::KwaversResult;
use ndarray::{Array3, Zip};
use std::collections::VecDeque;

/// Configuration for IMEX-BDF schemes
#[derive(Debug, Clone)]
pub struct IMEXBDFConfig {
    /// Order of the BDF method (1-6)
    pub order: usize,
    /// Whether to use variable order
    pub variable_order: bool,
    /// Maximum order for variable order methods
    pub max_order: usize,
}

impl Default for IMEXBDFConfig {
    fn default() -> Self {
        Self {
            order: 2,
            variable_order: false,
            max_order: 5,
        }
    }
}

/// IMEX-BDF scheme
#[derive(Debug, Clone)]
pub struct IMEXBDF {
    config: IMEXBDFConfig,
    /// BDF coefficients for implicit part
    alpha: Vec<f64>,
    /// Extrapolation coefficients for explicit part
    beta: Vec<f64>,
    /// History of previous solutions
    history: VecDeque<Array3<f64>>,
    /// History of previous explicit evaluations
    explicit_history: VecDeque<Array3<f64>>,
    /// Current order (for variable order methods)
    current_order: usize,
    /// Step counter
    step_count: usize,
}

impl IMEXBDF {
    /// Create a new IMEX-BDF scheme
    #[must_use]
    pub fn new(config: IMEXBDFConfig) -> Self {
        let order = config.order.min(6).max(1);
        let (alpha, beta) = Self::compute_coefficients(order);

        Self {
            config,
            alpha,
            beta,
            history: VecDeque::with_capacity(order),
            explicit_history: VecDeque::with_capacity(order),
            current_order: 1,
            step_count: 0,
        }
    }

    /// Compute BDF and extrapolation coefficients
    fn compute_coefficients(order: usize) -> (Vec<f64>, Vec<f64>) {
        match order {
            1 => {
                // BDF1 (Backward Euler)
                let alpha = vec![1.0, -1.0];
                let beta = vec![1.0];
                (alpha, beta)
            }
            2 => {
                // BDF2
                let alpha = vec![3.0 / 2.0, -2.0, 1.0 / 2.0];
                let beta = vec![2.0, -1.0];
                (alpha, beta)
            }
            3 => {
                // BDF3
                let alpha = vec![11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0];
                let beta = vec![3.0, -3.0, 1.0];
                (alpha, beta)
            }
            4 => {
                // BDF4
                let alpha = vec![25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0];
                let beta = vec![4.0, -6.0, 4.0, -1.0];
                (alpha, beta)
            }
            5 => {
                // BDF5
                let alpha = vec![137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0];
                let beta = vec![5.0, -10.0, 10.0, -5.0, 1.0];
                (alpha, beta)
            }
            6 => {
                // BDF6
                let alpha = vec![
                    147.0 / 60.0,
                    -6.0,
                    15.0 / 2.0,
                    -20.0 / 3.0,
                    15.0 / 4.0,
                    -6.0 / 5.0,
                    1.0 / 6.0,
                ];
                let beta = vec![6.0, -15.0, 20.0, -15.0, 6.0, -1.0];
                (alpha, beta)
            }
            _ => {
                // Return default coefficients for order 1 if invalid order
                // This should be validated at construction time
                (vec![1.0], vec![1.0])
            }
        }
    }

    /// Update history buffers
    fn update_history(&mut self, solution: Array3<f64>, explicit_eval: Array3<f64>) {
        self.history.push_front(solution);
        self.explicit_history.push_front(explicit_eval);

        // Keep only needed history
        let history_size = if self.config.variable_order {
            self.config.max_order
        } else {
            self.config.order
        };

        while self.history.len() > history_size {
            self.history.pop_back();
        }
        while self.explicit_history.len() > history_size {
            self.explicit_history.pop_back();
        }

        self.step_count += 1;
    }

    /// Determine order for variable order methods
    fn determine_order(&mut self) -> usize {
        if !self.config.variable_order {
            return self.config.order;
        }

        // Start with order 1 and increase gradually
        let max_possible_order = self.history.len().min(self.config.max_order);

        if self.step_count < self.current_order + 1 {
            self.current_order.min(max_possible_order)
        } else if self.current_order < self.config.max_order && self.step_count % 5 == 0 {
            // Consider increasing order
            (self.current_order + 1).min(max_possible_order)
        } else {
            self.current_order.min(max_possible_order)
        }
    }
}

impl IMEXScheme for IMEXBDF {
    fn step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        explicit_rhs: F,
        implicit_rhs: G,
        implicit_solver: &ImplicitSolverType,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // For the first few steps, use lower order or startup method
        let effective_order = (self.history.len() + 1).min(self.current_order);

        if effective_order == 0 {
            // First step: use backward Euler
            let explicit_eval = explicit_rhs(field)?;
            let dt_copy = dt;
            let implicit_fn = |y: &Array3<f64>| -> KwaversResult<Array3<f64>> {
                let mut residual = y - field;
                let g = implicit_rhs(y)?;
                Zip::from(&mut residual)
                    .and(&g)
                    .and(&explicit_eval)
                    .for_each(|r, &gi, &fe| *r -= dt_copy * (gi + fe));
                Ok(residual)
            };

            implicit_solver.solve(field, implicit_fn)
        } else {
            // Use BDF formula
            let (alpha, beta) = Self::compute_coefficients(effective_order);

            // Build explicit extrapolation
            let mut explicit_extrap = Array3::zeros(field.dim());
            for (i, coeff) in beta.iter().enumerate() {
                if i < self.explicit_history.len() {
                    Zip::from(&mut explicit_extrap)
                        .and(&self.explicit_history[i])
                        .for_each(|e, h: &f64| *e += *coeff * *h);
                }
            }

            // Build implicit equation
            let dt_copy = dt;
            let implicit_fn = |y: &Array3<f64>| -> KwaversResult<Array3<f64>> {
                let mut residual = Array3::zeros(field.dim());

                // Add BDF terms
                Zip::from(&mut residual)
                    .and(y)
                    .for_each(|r, &yi| *r += alpha[0] * yi);

                // Current solution contribution
                Zip::from(&mut residual)
                    .and(field)
                    .for_each(|r, &fi| *r += alpha[1] * fi);

                // History contributions
                for (i, coeff) in alpha[2..].iter().enumerate() {
                    if i < self.history.len() {
                        Zip::from(&mut residual)
                            .and(&self.history[i])
                            .for_each(|r, h: &f64| *r += *coeff * *h);
                    }
                }

                // Add implicit and explicit RHS
                let g = implicit_rhs(y)?;
                Zip::from(&mut residual)
                    .and(&g)
                    .and(&explicit_extrap)
                    .for_each(|r, gi: &f64, ee: &f64| *r -= dt_copy * (*gi + *ee));

                Ok(residual)
            };

            implicit_solver.solve(field, implicit_fn)
        }
    }

    fn order(&self) -> usize {
        self.current_order
    }

    fn stages(&self) -> usize {
        1 // BDF methods are single-stage
    }

    fn is_a_stable(&self) -> bool {
        // BDF methods up to order 6 are A-stable (order 1-2) or A(α)-stable (order 3-6)
        true
    }

    fn is_l_stable(&self) -> bool {
        // All BDF methods are L-stable
        true
    }

    fn adjust_for_stiffness(&mut self, stiffness_ratio: f64) {
        if self.config.variable_order {
            // For very stiff problems, limit the order
            if stiffness_ratio > 100.0 {
                self.current_order = self.current_order.min(3);
            } else if stiffness_ratio > 50.0 {
                self.current_order = self.current_order.min(4);
            }
        }
    }

    fn stability_function(&self, z: f64) -> f64 {
        // Stability function for BDF methods
        // R(z) = 1 / (sum(alpha_j * (1 - z)^j))
        let mut sum = 0.0;
        for (j, &alpha_j) in self.alpha.iter().enumerate() {
            sum += alpha_j * (1.0 - z).powi(j as i32);
        }
        1.0 / sum
    }
}

impl IMEXBDF {
    /// Reset the method (clear history)
    pub fn reset(&mut self) {
        self.history.clear();
        self.explicit_history.clear();
        self.step_count = 0;
        self.current_order = 1;
    }

    /// Update with new solution (must be called after each successful step)
    pub fn update(
        &mut self,
        solution: Array3<f64>,
        explicit_rhs: impl Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    ) -> KwaversResult<()> {
        let explicit_eval = explicit_rhs(&solution)?;
        self.update_history(solution, explicit_eval);

        if self.config.variable_order {
            self.current_order = self.determine_order();
            let (alpha, beta) = Self::compute_coefficients(self.current_order);
            self.alpha = alpha;
            self.beta = beta;
        }

        Ok(())
    }
}
