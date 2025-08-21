//! Adaptive time stepping
//!
//! This module provides adaptive time step control based on
//! error estimation and tolerance requirements.

use super::traits::{ErrorEstimatorTrait, TimeStepper};
use ndarray::{Array3, Zip};

/// Adaptive time stepper wrapper
///
/// Wraps any time stepper to provide adaptive time step control
#[derive(Debug)]
pub struct AdaptiveTimeStepper<T: TimeStepper> {
    /// Base time stepper
    base_stepper: T,
    /// Lower-order stepper for error estimation
    low_order_stepper: T,
    /// Error estimator
    error_estimator: Box<dyn ErrorEstimatorTrait>,
    /// Current time step
    current_dt: f64,
    /// Minimum allowed time step
    min_dt: f64,
    /// Maximum allowed time step
    max_dt: f64,
    /// Target tolerance
    tolerance: f64,
}

impl<T: TimeStepper> AdaptiveTimeStepper<T> {
    /// Create a new adaptive time stepper
    pub fn new(
        base_stepper: T,
        low_order_stepper: T,
        error_estimator: Box<dyn ErrorEstimatorTrait>,
        initial_dt: f64,
        min_dt: f64,
        max_dt: f64,
        tolerance: f64,
    ) -> Self {
        Self {
            base_stepper,
            low_order_stepper,
            error_estimator,
            current_dt: initial_dt,
            min_dt,
            max_dt,
            tolerance,
        }
    }

    /// Perform adaptive time step with error control
    pub fn adaptive_step<F>(
        &mut self,
        field: &Array3<f64>,
        rhs_fn: F,
        grid: &crate::grid::Grid,
    ) -> crate::KwaversResult<(Array3<f64>, f64)>
    where
        F: Fn(&Array3<f64>) -> crate::KwaversResult<Array3<f64>> + Clone,
    {
        let mut dt = self.current_dt;
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 10;

        loop {
            attempts += 1;
            if attempts > MAX_ATTEMPTS {
                return Err(crate::error::KwaversError::Numerical(
                    crate::error::NumericalError::Instability {
                        operation: "adaptive_step".to_string(),
                        condition: attempts as f64,
                    },
                ));
            }

            // Compute high-order solution (in-place update on a copy)
            let mut high_order = field.clone();
            self.base_stepper
                .step(&mut high_order, rhs_fn.clone(), dt, grid)?;

            // Compute low-order solution for error estimation (on another copy)
            let mut low_order = field.clone();
            self.low_order_stepper
                .step(&mut low_order, rhs_fn.clone(), dt, grid)?;

            // Estimate error
            let error = self
                .error_estimator
                .estimate_local_error(&low_order, &high_order, dt);

            // Check if error is acceptable
            if error <= self.tolerance {
                // Accept the step
                self.current_dt = self.compute_optimal_dt(dt, error);
                self.current_dt = self.current_dt.clamp(self.min_dt, self.max_dt);
                return Ok((high_order, dt));
            }

            // Reject the step and try with smaller dt
            dt = self.compute_optimal_dt(dt, error);
            dt = dt.max(self.min_dt);

            if dt == self.min_dt && error > 10.0 * self.tolerance {
                // Error too large even with minimum time step
                log::warn!(
                    "Adaptive time stepping: error {} exceeds tolerance {} even at min dt",
                    error,
                    self.tolerance
                );
            }
        }
    }

    /// Compute optimal time step based on error
    fn compute_optimal_dt(&self, current_dt: f64, error: f64) -> f64 {
        let safety_factor = 0.9;
        let max_increase = 2.0;
        let max_decrease = 0.1;

        if error < 1e-10 {
            // Error is essentially zero, increase time step
            return current_dt * max_increase;
        }

        let order = 4.0; // Assuming 4th order method
        let factor = safety_factor * (self.tolerance / error).powf(1.0 / (order + 1.0));
        let factor = factor.clamp(max_decrease, max_increase);

        current_dt * factor
    }

    /// Get current time step
    pub fn get_current_dt(&self) -> f64 {
        self.current_dt
    }
}

/// Richardson extrapolation error estimator
#[derive(Debug)]
pub struct RichardsonErrorEstimator {
    /// Order of the method
    order: usize,
}

impl RichardsonErrorEstimator {
    /// Create a new Richardson error estimator
    pub fn new(order: usize) -> Self {
        Self { order }
    }
}

impl ErrorEstimatorTrait for RichardsonErrorEstimator {
    fn estimate_local_error(
        &self,
        field_low: &Array3<f64>,
        field_high: &Array3<f64>,
        _dt: f64,
    ) -> f64 {
        // Richardson extrapolation error estimate
        let mut max_error: f64 = 0.0;

        Zip::from(field_low)
            .and(field_high)
            .for_each(|&low, &high| {
                let error = (high - low).abs();
                max_error = max_error.max(error);
            });

        // Scale by order-dependent factor
        let factor = 1.0 / (2.0_f64.powi(self.order as i32) - 1.0);
        max_error * factor
    }

    fn estimate_global_error(&self, local_errors: &[f64], time_steps: &[f64]) -> f64 {
        // Estimate global error accumulation
        if local_errors.is_empty() || time_steps.is_empty() {
            return 0.0;
        }

        // Simple accumulation model
        let mut global_error = 0.0;
        for (i, (&local_err, &dt)) in local_errors.iter().zip(time_steps.iter()).enumerate() {
            // Error growth model: e_global ~ sum(e_local * sqrt(n))
            global_error += local_err * dt * ((i + 1) as f64).sqrt();
        }

        global_error
    }
}

/// Embedded Runge-Kutta error estimator
#[derive(Debug)]
pub struct EmbeddedRKErrorEstimator {
    /// Norm type for error computation
    norm_type: ErrorNorm,
}

/// Error norm types
#[derive(Debug, Clone, Copy)]
pub enum ErrorNorm {
    /// L-infinity norm (maximum)
    LInfinity,
    /// L2 norm (RMS)
    L2,
    /// L1 norm (average)
    L1,
}

impl EmbeddedRKErrorEstimator {
    /// Create a new embedded RK error estimator
    pub fn new(norm_type: ErrorNorm) -> Self {
        Self { norm_type }
    }
}

impl ErrorEstimatorTrait for EmbeddedRKErrorEstimator {
    fn estimate_local_error(
        &self,
        field_low: &Array3<f64>,
        field_high: &Array3<f64>,
        _dt: f64,
    ) -> f64 {
        match self.norm_type {
            ErrorNorm::LInfinity => {
                let mut max_error: f64 = 0.0;
                Zip::from(field_low)
                    .and(field_high)
                    .for_each(|&low, &high| {
                        max_error = max_error.max((high - low).abs());
                    });
                max_error
            }
            ErrorNorm::L2 => {
                let mut sum_sq = 0.0;
                let mut count = 0;
                Zip::from(field_low)
                    .and(field_high)
                    .for_each(|&low, &high| {
                        sum_sq += (high - low).powi(2);
                        count += 1;
                    });
                (sum_sq / count as f64).sqrt()
            }
            ErrorNorm::L1 => {
                let mut sum = 0.0;
                let mut count = 0;
                Zip::from(field_low)
                    .and(field_high)
                    .for_each(|&low, &high| {
                        sum += (high - low).abs();
                        count += 1;
                    });
                sum / count as f64
            }
        }
    }

    fn estimate_global_error(&self, local_errors: &[f64], _time_steps: &[f64]) -> f64 {
        // Simple sum for embedded methods
        local_errors.iter().sum()
    }
}

/// Default error estimator
pub type ErrorEstimator = RichardsonErrorEstimator;
