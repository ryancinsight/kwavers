//! Time stepper implementations
//!
//! This module provides various time integration methods including
//! Runge-Kutta and Adams-Bashforth schemes.

use super::traits::{TimeStepper, TimeStepperConfig};
use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::{Array3, Zip};
use std::collections::VecDeque;

/// Configuration for Runge-Kutta 4 method
#[derive(Debug, Clone)]
pub struct RK4Config {
    /// Stability safety factor
    pub safety_factor: f64,
}

impl Default for RK4Config {
    fn default() -> Self {
        Self { safety_factor: 0.9 }
    }
}

impl TimeStepperConfig for RK4Config {
    fn order(&self) -> usize {
        4
    }
    fn stages(&self) -> usize {
        4
    }
    fn is_explicit(&self) -> bool {
        true
    }
    fn validate(&self) -> KwaversResult<()> {
        Ok(())
    }
}

/// 4th-order Runge-Kutta time stepper
#[derive(Debug, Debug)]
pub struct RungeKutta4 {
    config: RK4Config,
    /// Temporary storage for intermediate stages
    k1: Option<Array3<f64>>,
    k2: Option<Array3<f64>>,
    k3: Option<Array3<f64>>,
    k4: Option<Array3<f64>>,
    /// Workspace for intermediate field values (avoids allocations)
    intermediate_field: Option<Array3<f64>>,
}

impl TimeStepper for RungeKutta4 {
    type Config = RK4Config;

    fn new(config: Self::Config) -> Self {
        Self {
            config,
            k1: None,
            k2: None,
            k3: None,
            k4: None,
            intermediate_field: None,
        }
    }

    fn step<F>(
        &mut self,
        field: &mut Array3<f64>,
        rhs_fn: F,
        dt: f64,
        _grid: &Grid,
    ) -> KwaversResult<()>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let shape = field.dim();

        // Initialize storage if needed (lazy initialization)
        if self.k1.is_none() {
            self.k1 = Some(Array3::zeros(shape));
            self.k2 = Some(Array3::zeros(shape));
            self.k3 = Some(Array3::zeros(shape));
            self.k4 = Some(Array3::zeros(shape));
            self.intermediate_field = Some(Array3::zeros(shape));
        }

        let k1 = self.k1.as_mut().unwrap();
        let k2 = self.k2.as_mut().unwrap();
        let k3 = self.k3.as_mut().unwrap();
        let k4 = self.k4.as_mut().unwrap();
        let intermediate_field = self.intermediate_field.as_mut().unwrap();

        // Stage 1: k1 = f(t, y)
        *k1 = rhs_fn(field)?;

        // Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        // Use pre-allocated workspace instead of cloning
        intermediate_field.assign(field);
        Zip::from(&mut *intermediate_field)
            .and(&*k1)
            .for_each(|t, k| *t += 0.5 * dt * *k);
        *k2 = rhs_fn(intermediate_field)?;

        // Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        // Reuse the same workspace
        intermediate_field.assign(field);
        Zip::from(&mut *intermediate_field)
            .and(&*k2)
            .for_each(|t, k| *t += 0.5 * dt * *k);
        *k3 = rhs_fn(intermediate_field)?;

        // Stage 4: k4 = f(t + dt, y + dt * k3)
        intermediate_field.assign(field);
        Zip::from(&mut *intermediate_field)
            .and(&*k3)
            .for_each(|t, k| *t += dt * *k);
        *k4 = rhs_fn(intermediate_field)?;

        // Combine stages: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        // Update field in-place
        Zip::from(field)
            .and(&*k1)
            .and(&*k2)
            .and(&*k3)
            .and(&*k4)
            .for_each(|r, k1, k2, k3, k4| {
                *r += dt / 6.0 * (*k1 + 2.0 * *k2 + 2.0 * *k3 + *k4);
            });

        Ok(())
    }

    fn stability_factor(&self) -> f64 {
        2.8 * self.config.safety_factor
    }
}

/// Configuration for Adams-Bashforth methods
#[derive(Debug, Clone)]
pub struct AdamsBashforthConfig {
    /// Order of the method (2 or 3)
    pub order: usize,
    /// Number of startup steps using RK4
    pub startup_steps: usize,
}

impl Default for AdamsBashforthConfig {
    fn default() -> Self {
        let order = 2;
        Self {
            order,
            // The number of startup steps should be order - 1
            startup_steps: order.saturating_sub(1),
        }
    }
}

impl TimeStepperConfig for AdamsBashforthConfig {
    fn order(&self) -> usize {
        self.order
    }
    fn stages(&self) -> usize {
        1
    }
    fn is_explicit(&self) -> bool {
        true
    }

    fn validate(&self) -> KwaversResult<()> {
        if self.order != 2 && self.order != 3 {
            return Err(crate::error::KwaversError::Validation(
                crate::error::ValidationError::FieldValidation {
                    field: "order".to_string(),
                    value: self.order.to_string(),
                    constraint: "Must be 2 or 3".to_string(),
                },
            ));
        }
        Ok(())
    }
}

/// Adams-Bashforth multi-step time stepper
///
/// # Memory Usage
///
/// This is a multi-step method that stores the results of previous RHS
/// evaluations. The number of historical fields stored is equal to the
/// order of the method. For large grids, this can result in significant
/// memory consumption. For example, a 3rd-order scheme will store 3
/// historical fields, which can consume several gigabytes of RAM for a
/// high-resolution 3D grid (e.g., 512^3). Consider using a single-step
/// method like `RungeKutta4` if memory is a concern.
#[derive(Debug, Debug)]
pub struct AdamsBashforth {
    config: AdamsBashforthConfig,
    /// History of previous RHS evaluations
    rhs_history: VecDeque<Array3<f64>>,
    /// Startup stepper (RK4)
    startup_stepper: RungeKutta4,
    /// Number of steps taken
    step_count: usize,
}

impl TimeStepper for AdamsBashforth {
    type Config = AdamsBashforthConfig;

    fn new(config: Self::Config) -> Self {
        // Validate that we have enough startup steps
        assert!(
            config.startup_steps >= config.order.saturating_sub(1),
            "Not enough startup steps for the requested Adams-Bashforth order. \
             Need at least {} startup steps for order {}",
            config.order - 1,
            config.order
        );

        let history_size = config.order;
        Self {
            config,
            rhs_history: VecDeque::with_capacity(history_size),
            startup_stepper: RungeKutta4::new(RK4Config::default()),
            step_count: 0,
        }
    }

    fn step<F>(
        &mut self,
        field: &mut Array3<f64>,
        rhs_fn: F,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // Use RK4 for startup
        if self.step_count < self.config.startup_steps {
            self.step_count += 1;

            // Store RHS evaluation before updating field
            let rhs = rhs_fn(field)?;

            // Update field in-place using RK4
            self.startup_stepper.step(field, &rhs_fn, dt, grid)?;

            // Store RHS in history
            self.rhs_history.push_back(rhs);
            if self.rhs_history.len() > self.config.order {
                self.rhs_history.pop_front();
            }

            return Ok(());
        }

        // Evaluate RHS at current state
        let current_rhs = rhs_fn(field)?;

        // Apply Adams-Bashforth formula in-place
        match self.config.order {
            2 => {
                // AB2: y_{n+1} = y_n + dt * (3/2 * f_n - 1/2 * f_{n-1})
                if !self.rhs_history.is_empty() {
                    let f_n = &current_rhs;
                    let f_nm1 = &self.rhs_history[self.rhs_history.len() - 1];

                    Zip::from(field)
                        .and(f_n)
                        .and(f_nm1)
                        .for_each(|r, fn_val, fnm1_val| {
                            *r += dt * (1.5 * *fn_val - 0.5 * *fnm1_val);
                        });
                }
            }
            3 => {
                // AB3: y_{n+1} = y_n + dt * (23/12 * f_n - 16/12 * f_{n-1} + 5/12 * f_{n-2})
                if self.rhs_history.len() >= 2 {
                    let f_n = &current_rhs;
                    let f_nm1 = &self.rhs_history[self.rhs_history.len() - 1];
                    let f_nm2 = &self.rhs_history[self.rhs_history.len() - 2];

                    Zip::from(field).and(f_n).and(f_nm1).and(f_nm2).for_each(
                        |r, fn_val, fnm1_val, fnm2_val| {
                            *r += dt
                                * (23.0 / 12.0 * *fn_val - 16.0 / 12.0 * *fnm1_val
                                    + 5.0 / 12.0 * *fnm2_val);
                        },
                    );
                }
            }
            _ => {
                return Err(crate::KwaversError::Config(
                    crate::ConfigError::InvalidValue {
                        parameter: "order".to_string(),
                        value: self.config.order.to_string(),
                        constraint: "1, 2, 3, or 4".to_string(),
                    },
                ));
            }
        }

        // Update history
        self.rhs_history.push_back(current_rhs);
        if self.rhs_history.len() > self.config.order {
            self.rhs_history.pop_front();
        }

        self.step_count += 1;
        Ok(())
    }

    fn stability_factor(&self) -> f64 {
        match self.config.order {
            2 => 1.0,
            3 => 0.5,
            _ => 0.5,
        }
    }

    fn reset(&mut self) {
        self.rhs_history.clear();
        self.step_count = 0;
    }
}
