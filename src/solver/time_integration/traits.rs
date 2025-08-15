//! Traits for time integration
//! 
//! This module defines the common interfaces for time integration methods,
//! following the Interface Segregation Principle (ISP).

use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::Array3;
use std::fmt::Debug;

/// Configuration trait for time steppers
pub trait TimeStepperConfig: Clone + Send + Sync + Debug {
    /// Get the order of accuracy
    fn order(&self) -> usize;
    
    /// Get the number of stages (for multi-stage methods)
    fn stages(&self) -> usize;
    
    /// Is this an explicit method?
    fn is_explicit(&self) -> bool;
    
    /// Validate the configuration
    fn validate(&self) -> KwaversResult<()>;
}

/// Base trait for time stepping methods
pub trait TimeStepper: Send + Sync + Debug {
    /// Configuration type for this time stepper
    type Config: TimeStepperConfig;
    
    /// Create a new time stepper with given configuration
    fn new(config: Self::Config) -> Self;
    
    /// Advance the solution by one time step (in-place)
    /// 
    /// # Arguments
    /// * `field` - Current field values (will be updated in-place)
    /// * `rhs_fn` - Function that computes the right-hand side (time derivative)
    /// * `dt` - Time step size
    /// * `grid` - Computational grid
    fn step<F>(
        &mut self,
        field: &mut Array3<f64>,
        rhs_fn: F,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>;
    
    /// Get the stability region scaling factor
    fn stability_factor(&self) -> f64;
    
    /// Reset internal state (for multi-step methods)
    fn reset(&mut self) {}
}

/// Trait for adaptive time stepping
pub trait AdaptiveTimeStepperTrait: TimeStepper {
    /// Estimate the error for adaptive time stepping
    fn estimate_error(
        &self,
        field: &Array3<f64>,
        updated_field: &Array3<f64>,
        dt: f64,
    ) -> f64;
    
    /// Compute optimal time step based on error estimate
    fn compute_optimal_dt(
        &self,
        current_dt: f64,
        error: f64,
        tolerance: f64,
    ) -> f64 {
        // Default implementation using standard formula
        let safety_factor = 0.9;
        let max_increase = 2.0;
        let max_decrease = 0.1;
        
        let factor = safety_factor * (tolerance / error).powf(1.0 / (self.order() as f64 + 1.0));
        let factor = factor.clamp(max_decrease, max_increase);
        
        current_dt * factor
    }
    
    /// Get the order of the method for error estimation
    fn order(&self) -> usize;
}

/// Configuration for multi-rate time integration
#[derive(Debug, Clone)]
pub struct MultiRateConfig {
    /// Maximum number of subcycles allowed
    pub max_subcycles: usize,
    /// Stability safety factor (0 < factor <= 1)
    pub stability_factor: f64,
    /// Tolerance for adaptive time stepping
    pub adaptive_tolerance: f64,
    /// Minimum allowed time step
    pub min_dt: f64,
    /// Maximum allowed time step
    pub max_dt: f64,
    /// Enable adaptive time stepping
    pub adaptive: bool,
    /// Time stepper type for each component
    pub time_steppers: std::collections::HashMap<String, TimeStepperType>,
    /// CFL safety factor for time step computation
    pub cfl_safety_factor: f64,
}

impl Default for MultiRateConfig {
    fn default() -> Self {
        Self {
            max_subcycles: 10,
            stability_factor: 0.9,
            adaptive_tolerance: 1e-6,
            min_dt: 1e-10,
            max_dt: 1.0,
            adaptive: true,
            time_steppers: std::collections::HashMap::new(),
            cfl_safety_factor: 0.9,
        }
    }
}

/// Available time stepper types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeStepperType {
    /// 4th-order Runge-Kutta
    RungeKutta4,
    /// 2nd-order Adams-Bashforth
    AdamsBashforth2,
    /// 3rd-order Adams-Bashforth
    AdamsBashforth3,
    /// Forward Euler (1st order)
    ForwardEuler,
    /// Leapfrog (2nd order)
    Leapfrog,
}

impl TimeStepperType {
    /// Get the order of accuracy
    pub fn order(&self) -> usize {
        match self {
            Self::RungeKutta4 => 4,
            Self::AdamsBashforth2 => 2,
            Self::AdamsBashforth3 => 3,
            Self::ForwardEuler => 1,
            Self::Leapfrog => 2,
        }
    }
    
    /// Get the stability factor for CFL calculation
    pub fn stability_factor(&self) -> f64 {
        match self {
            Self::RungeKutta4 => 2.8,  // Approximate stability limit
            Self::AdamsBashforth2 => 1.0,
            Self::AdamsBashforth3 => 0.5,
            Self::ForwardEuler => 1.0,
            Self::Leapfrog => 1.0,
        }
    }
}

/// Trait for error estimation in time integration
pub trait ErrorEstimatorTrait: Send + Sync + Debug {
    /// Estimate the local truncation error
    fn estimate_local_error(
        &self,
        field_low: &Array3<f64>,
        field_high: &Array3<f64>,
        dt: f64,
    ) -> f64;
    
    /// Estimate the global error accumulation
    fn estimate_global_error(
        &self,
        local_errors: &[f64],
        time_steps: &[f64],
    ) -> f64;
}