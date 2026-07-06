//! Time stepper implementations
//!
//! This module provides various time integration methods including
//! Runge-Kutta and Adams-Bashforth schemes.

use super::traits::{TimeStepper, TimeStepperConfig};
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, SystemError};
use kwavers_grid::Grid;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{Array3, Zip};
use std::collections::VecDeque;

fn add_scaled_inplace(target: &mut Array3<f64>, rhs: &Array3<f64>, scale: f64) {
    assert_eq!(
        target.dim(),
        rhs.dim(),
        "invariant: add_scaled_inplace shape mismatch"
    );
    match (
        target.as_slice_memory_order_mut(),
        rhs.as_slice_memory_order(),
    ) {
        (Some(target_slice), Some(rhs_slice)) => {
            enumerate_mut_with::<Adaptive, _, _>(target_slice, |idx, value| {
                *value += scale * rhs_slice[idx];
            });
        }
        _ => Zip::from(target)
            .and(rhs)
            .for_each(|value, &rhs| *value += scale * rhs),
    }
}

fn combine_rk4_inplace(
    field: &mut Array3<f64>,
    k1: &Array3<f64>,
    k2: &Array3<f64>,
    k3: &Array3<f64>,
    k4: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(field.dim(), k1.dim(), "invariant: RK4 k1 shape mismatch");
    assert_eq!(field.dim(), k2.dim(), "invariant: RK4 k2 shape mismatch");
    assert_eq!(field.dim(), k3.dim(), "invariant: RK4 k3 shape mismatch");
    assert_eq!(field.dim(), k4.dim(), "invariant: RK4 k4 shape mismatch");
    match (
        field.as_slice_memory_order_mut(),
        k1.as_slice_memory_order(),
        k2.as_slice_memory_order(),
        k3.as_slice_memory_order(),
        k4.as_slice_memory_order(),
    ) {
        (Some(field_slice), Some(k1_slice), Some(k2_slice), Some(k3_slice), Some(k4_slice)) => {
            enumerate_mut_with::<Adaptive, _, _>(field_slice, |idx, value| {
                *value += dt / 6.0
                    * (2.0f64.mul_add(k3_slice[idx], 2.0f64.mul_add(k2_slice[idx], k1_slice[idx]))
                        + k4_slice[idx]);
            });
        }
        _ => Zip::from(field)
            .and(k1)
            .and(k2)
            .and(k3)
            .and(k4)
            .for_each(|r, k1, k2, k3, k4| {
                *r += dt / 6.0 * (2.0f64.mul_add(*k3, 2.0f64.mul_add(*k2, *k1)) + *k4);
            }),
    }
}

fn adams_bashforth2_inplace(
    field: &mut Array3<f64>,
    f_n: &Array3<f64>,
    f_nm1: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(field.dim(), f_n.dim(), "invariant: AB2 f_n shape mismatch");
    assert_eq!(
        field.dim(),
        f_nm1.dim(),
        "invariant: AB2 f_nm1 shape mismatch"
    );
    match (
        field.as_slice_memory_order_mut(),
        f_n.as_slice_memory_order(),
        f_nm1.as_slice_memory_order(),
    ) {
        (Some(field_slice), Some(f_n_slice), Some(f_nm1_slice)) => {
            enumerate_mut_with::<Adaptive, _, _>(field_slice, |idx, value| {
                *value += dt * 1.5f64.mul_add(f_n_slice[idx], -(0.5 * f_nm1_slice[idx]));
            });
        }
        _ => Zip::from(field)
            .and(f_n)
            .and(f_nm1)
            .for_each(|r, fn_val, fnm1_val| {
                *r += dt * 1.5f64.mul_add(*fn_val, -(0.5 * *fnm1_val));
            }),
    }
}

fn adams_bashforth3_inplace(
    field: &mut Array3<f64>,
    f_n: &Array3<f64>,
    f_nm1: &Array3<f64>,
    f_nm2: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(field.dim(), f_n.dim(), "invariant: AB3 f_n shape mismatch");
    assert_eq!(
        field.dim(),
        f_nm1.dim(),
        "invariant: AB3 f_nm1 shape mismatch"
    );
    assert_eq!(
        field.dim(),
        f_nm2.dim(),
        "invariant: AB3 f_nm2 shape mismatch"
    );
    match (
        field.as_slice_memory_order_mut(),
        f_n.as_slice_memory_order(),
        f_nm1.as_slice_memory_order(),
        f_nm2.as_slice_memory_order(),
    ) {
        (Some(field_slice), Some(f_n_slice), Some(f_nm1_slice), Some(f_nm2_slice)) => {
            enumerate_mut_with::<Adaptive, _, _>(field_slice, |idx, value| {
                *value += dt
                    * (5.0_f64 / 12.0).mul_add(
                        f_nm2_slice[idx],
                        (23.0_f64 / 12.0)
                            .mul_add(f_n_slice[idx], -(16.0_f64 / 12.0 * f_nm1_slice[idx])),
                    );
            });
        }
        _ => Zip::from(field).and(f_n).and(f_nm1).and(f_nm2).for_each(
            |r, fn_val, fnm1_val, fnm2_val| {
                *r += dt
                    * (5.0_f64 / 12.0).mul_add(
                        *fnm2_val,
                        (23.0_f64 / 12.0).mul_add(*fn_val, -(16.0_f64 / 12.0 * *fnm1_val)),
                    );
            },
        ),
    }
}

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
#[derive(Debug)]
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

        // Safe access to pre-allocated storage
        let k1 = self.k1.as_mut().ok_or_else(|| {
            KwaversError::System(SystemError::ResourceExhausted {
                resource: "RK4 k1 storage".to_owned(),
                reason: "Storage not initialized".to_owned(),
            })
        })?;
        let k2 = self.k2.as_mut().ok_or_else(|| {
            KwaversError::System(SystemError::ResourceExhausted {
                resource: "RK4 k2 storage".to_owned(),
                reason: "Storage not initialized".to_owned(),
            })
        })?;
        let k3 = self.k3.as_mut().ok_or_else(|| {
            KwaversError::System(SystemError::ResourceExhausted {
                resource: "RK4 k3 storage".to_owned(),
                reason: "Storage not initialized".to_owned(),
            })
        })?;
        let k4 = self.k4.as_mut().ok_or_else(|| {
            KwaversError::System(SystemError::ResourceExhausted {
                resource: "RK4 k4 storage".to_owned(),
                reason: "Storage not initialized".to_owned(),
            })
        })?;
        let intermediate_field = self.intermediate_field.as_mut().ok_or_else(|| {
            KwaversError::System(SystemError::ResourceExhausted {
                resource: "RK4 intermediate field storage".to_owned(),
                reason: "Storage not initialized".to_owned(),
            })
        })?;

        // Stage 1: k1 = f(t, y)
        *k1 = rhs_fn(field)?;

        // Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        // Use pre-allocated workspace instead of cloning
        intermediate_field.assign(field);
        add_scaled_inplace(intermediate_field, k1, 0.5 * dt);
        *k2 = rhs_fn(intermediate_field)?;

        // Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        // Reuse the same workspace
        intermediate_field.assign(field);
        add_scaled_inplace(intermediate_field, k2, 0.5 * dt);
        *k3 = rhs_fn(intermediate_field)?;

        // Stage 4: k4 = f(t + dt, y + dt * k3)
        intermediate_field.assign(field);
        add_scaled_inplace(intermediate_field, k3, dt);
        *k4 = rhs_fn(intermediate_field)?;

        // Combine stages: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        // Update field in-place
        combine_rk4_inplace(field, k1, k2, k3, k4, dt);

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
            return Err(kwavers_core::error::KwaversError::Validation(
                kwavers_core::error::ValidationError::FieldValidation {
                    field: "order".to_owned(),
                    value: self.order.to_string(),
                    constraint: "Must be 2 or 3".to_owned(),
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
#[derive(Debug)]
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

                    adams_bashforth2_inplace(field, f_n, f_nm1, dt);
                }
            }
            3 => {
                // AB3: y_{n+1} = y_n + dt * (23/12 * f_n - 16/12 * f_{n-1} + 5/12 * f_{n-2})
                if self.rhs_history.len() >= 2 {
                    let f_n = &current_rhs;
                    let f_nm1 = &self.rhs_history[self.rhs_history.len() - 1];
                    let f_nm2 = &self.rhs_history[self.rhs_history.len() - 2];

                    adams_bashforth3_inplace(field, f_n, f_nm1, f_nm2, dt);
                }
            }
            _ => {
                return Err(kwavers_core::error::KwaversError::Config(
                    kwavers_core::error::ConfigError::InvalidValue {
                        parameter: "order".to_owned(),
                        value: self.config.order.to_string(),
                        constraint: "1, 2, 3, or 4".to_owned(),
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
