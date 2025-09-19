//! Multi-Rate Time Integration Module
//!
//! This module implements multi-rate time integration methods that allow
//! different time steps for different physics components, enabling efficient
//! simulation of multi-physics problems with disparate time scales.
//!
//! # Design Principles
//! - SOLID: Separate modules for time steppers, stability analysis, and coupling
//! - CUPID: Composable time integration schemes with clear interfaces
//! - GRASP: Clear separation between time stepping logic and physics evaluation
//! - DRY: Shared utilities for time step computation and error estimation
//! - KISS: Simple interface for multi-rate time stepping
//! - YAGNI: Only implementing well-established multi-rate methods
//! - Clean: Comprehensive documentation and testing

pub mod adaptive_stepping;
pub mod conservation;
pub mod coupling;
pub mod multi_rate_controller;
pub mod stability;
pub mod time_scale_separation;
pub mod time_stepper;

/// Stability constraints for time stepping
#[derive(Debug, Clone, Default)]
pub struct StabilityConstraints {
    /// Maximum stable time step
    pub max_dt: Option<f64>,
    /// CFL number constraint
    pub cfl_number: Option<f64>,
}
pub mod traits;

// Re-export main types
pub use adaptive_stepping::{AdaptiveTimeStepper, ErrorEstimator};
pub use conservation::{ConservationError, ConservationMonitor, ConservedQuantities};
pub use coupling::{SubcyclingStrategy, TimeCoupling};
pub use multi_rate_controller::MultiRateController;
pub use stability::{CFLCondition, StabilityAnalyzer};
pub use time_scale_separation::{TimeScale, TimeScaleSeparator};
pub use time_stepper::{AdamsBashforth, RungeKutta4};
pub use traits::{MultiRateConfig, TimeStepper, TimeStepperConfig, TimeStepperType};

use crate::error::{KwaversError, ValidationError};
use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::Array3;
use std::collections::HashMap;

/// Multi-Rate Time Integration System
///
/// Manages different time steps for different physics components
/// while maintaining stability and accuracy.
#[derive(Debug)]
pub struct MultiRateTimeIntegrator {
    /// Configuration for multi-rate integration
    #[allow(dead_code)]
    config: MultiRateConfig,
    /// Controller for managing multiple time scales
    controller: MultiRateController,
    /// Stability analyzer for CFL conditions
    stability_analyzer: StabilityAnalyzer,
    /// Time coupling strategy
    coupling: Box<dyn TimeCoupling>,
    /// History of time steps for each component
    time_step_history: HashMap<String, Vec<f64>>,
    /// CFL safety factor
    cfl_safety_factor: f64,
    /// Time scale separator for automatic detection
    #[allow(dead_code)]
    time_scale_separator: TimeScaleSeparator,
    /// Conservation monitor
    #[allow(dead_code)]
    conservation_monitor: ConservationMonitor,
}

impl MultiRateTimeIntegrator {
    /// Create a new multi-rate time integrator
    pub fn new(config: MultiRateConfig, grid: &Grid) -> Self {
        let controller = MultiRateController::new(config.clone());
        let stability_analyzer = StabilityAnalyzer::new(config.stability_factor);
        let coupling = SubcyclingStrategy::new(config.max_subcycles);
        let time_scale_separator = TimeScaleSeparator::new(grid);
        let conservation_monitor = ConservationMonitor::new(grid);

        Self {
            config: config.clone(),
            controller,
            stability_analyzer,
            coupling: Box::new(coupling),
            time_step_history: HashMap::new(),
            cfl_safety_factor: config.cfl_safety_factor,
            time_scale_separator,
            conservation_monitor,
        }
    }

    /// Advance the solution using multi-rate time integration
    pub fn advance(
        &mut self,
        fields: &mut HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::physics::plugin::Plugin>>,
        global_time: f64,
        target_time: f64,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        // Validate inputs
        if target_time <= global_time {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "target_time".to_string(),
                value: target_time.to_string(),
                constraint: format!("Must be greater than global_time {global_time}"),
            }));
        }

        // Step 1: Compute stable time steps for each component
        let component_time_steps =
            self.compute_component_time_steps(fields, physics_components, grid)?;

        // Step 2: Determine global time step and subcycling strategy
        let (global_dt, subcycles) = self
            .controller
            .determine_time_steps(&component_time_steps, target_time - global_time)?;

        // Step 3: Perform multi-rate time integration
        let mut current_time = global_time;

        while current_time < target_time {
            let dt = global_dt.min(target_time - current_time);

            // Apply coupling strategy to advance all components
            self.coupling.advance_coupled_system(
                fields,
                physics_components,
                &subcycles,
                dt,
                grid,
            )?;

            current_time += dt;

            // Update time step history
            for (component, &local_dt) in &component_time_steps {
                self.time_step_history
                    .entry(component.clone())
                    .or_default()
                    .push(local_dt);
            }
        }

        Ok(current_time)
    }

    /// Compute stable time steps for each physics component
    fn compute_component_time_steps(
        &self,
        fields: &HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::physics::plugin::Plugin>>,
        grid: &Grid,
    ) -> KwaversResult<HashMap<String, f64>> {
        physics_components
            .iter()
            .map(|(name, component)| {
                // Get the field associated with this component
                let field = fields.get(name).ok_or_else(|| {
                    KwaversError::Validation(ValidationError::FieldValidation {
                        field: "fields".to_string(),
                        value: name.clone(),
                        constraint: "Field not found for component".to_string(),
                    })
                })?;

                // Get stability constraints from the component
                let _constraints = component.stability_constraints();

                // Compute CFL-limited time step
                let max_dt = self.stability_analyzer.compute_stable_dt_from_constraints(
                    field,
                    grid,
                    &std::collections::HashMap::new(),
                )?;

                Ok((name.clone(), max_dt * self.cfl_safety_factor))
            })
            .collect::<KwaversResult<HashMap<String, f64>>>()
    }

    /// Get time stepping statistics
    pub fn get_statistics(&self) -> TimeSteppingStatistics {
        TimeSteppingStatistics {
            total_steps: self.controller.total_steps(),
            subcycle_counts: self.controller.subcycle_counts(),
            average_time_steps: self.compute_average_time_steps(),
            efficiency_ratio: self.controller.efficiency_ratio(),
        }
    }

    /// Compute average time steps for each component
    fn compute_average_time_steps(&self) -> HashMap<String, f64> {
        self.time_step_history
            .iter()
            .map(|(name, history)| {
                let avg = if history.is_empty() {
                    0.0
                } else {
                    history.iter().sum::<f64>() / history.len() as f64
                };
                (name.clone(), avg)
            })
            .collect()
    }
}

/// Statistics for multi-rate time stepping
#[derive(Debug, Clone)]
pub struct TimeSteppingStatistics {
    /// Total number of global time steps
    pub total_steps: usize,
    /// Number of subcycles for each component
    pub subcycle_counts: HashMap<String, usize>,
    /// Average time step for each component
    pub average_time_steps: HashMap<String, f64>,
    /// Efficiency ratio (actual work vs single-rate)
    pub efficiency_ratio: f64,
}

#[cfg(test)]
mod tests;
