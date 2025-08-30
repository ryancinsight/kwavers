//! Multi-rate controller
//!
//! This module manages different time scales for multi-physics simulations.
//!
//! # Multi-Rate Time Integration
//!
//! Multi-rate integration allows different physics components to use different
//! time steps based on their stability requirements:
//! - Slow components (e.g., thermal diffusion) take large, efficient steps
//! - Fast components (e.g., acoustic waves) take multiple smaller sub-steps
//!
//! The global time step is set by the SLOWEST component to maximize efficiency.

use super::traits::MultiRateConfig;
use crate::error::{KwaversError, ValidationError};
use crate::KwaversResult;
use log::{debug, info};
use std::collections::HashMap;

/// Controller for multi-rate time integration
#[derive(Debug, Debug)]
pub struct MultiRateController {
    config: MultiRateConfig,
    /// Total number of global steps taken
    total_steps: usize,
    /// Subcycle counts for each component
    subcycle_counts: HashMap<String, usize>,
}

impl MultiRateController {
    /// Create a new multi-rate controller
    pub fn new(config: MultiRateConfig) -> Self {
        Self {
            config,
            total_steps: 0,
            subcycle_counts: HashMap::new(),
        }
    }

    /// Determine global time step and subcycling strategy
    ///
    /// The global time step is set to the MAXIMUM stable time step among all
    /// components (i.e., the slowest component's time step). Faster components
    /// then take multiple sub-steps within each global step.
    pub fn determine_time_steps(
        &mut self,
        component_time_steps: &HashMap<String, f64>,
        max_dt: f64,
    ) -> KwaversResult<(f64, HashMap<String, usize>)> {
        if component_time_steps.is_empty() {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "component_time_steps".to_string(),
                value: "empty".to_string(),
                constraint: "Must have at least one component".to_string(),
            }));
        }

        // Find the MAXIMUM time step (slowest component) for the global step
        // This maximizes efficiency by letting slow components take large steps
        let global_dt = component_time_steps
            .values()
            .cloned()
            .fold(0.0, f64::max) // Use maximum, not minimum!
            .min(max_dt) // Still respect the overall maximum
            .max(self.config.min_dt); // And the minimum

        debug!("Multi-rate: Global dt = {} (slowest component)", global_dt);

        // Compute subcycles for each component
        // Faster components need MORE subcycles within the global step
        let mut subcycles = HashMap::new();

        for (name, &component_dt) in component_time_steps {
            // Calculate how many sub-steps this component needs
            // within the global time step
            let n_subcycles = if component_dt >= global_dt {
                // This is a slow component - takes one step per global step
                1
            } else {
                // This is a fast component - needs multiple sub-steps
                // n = ceil(global_dt / component_dt)
                ((global_dt / component_dt).ceil() as usize)
                    .max(1)
                    .min(self.config.max_subcycles)
            };

            debug!(
                "  Component '{}': dt={}, subcycles={}",
                name, component_dt, n_subcycles
            );

            subcycles.insert(name.clone(), n_subcycles);

            // Update statistics
            *self.subcycle_counts.entry(name.clone()).or_insert(0) += n_subcycles;
        }

        self.total_steps += 1;

        // Log efficiency gain
        if self.total_steps % 100 == 0 {
            info!(
                "Multi-rate efficiency ratio: {:.2}x",
                self.efficiency_ratio()
            );
        }

        Ok((global_dt, subcycles))
    }

    /// Get total number of steps
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get subcycle counts
    pub fn subcycle_counts(&self) -> HashMap<String, usize> {
        self.subcycle_counts.clone()
    }

    /// Compute efficiency ratio
    ///
    /// Returns the ratio of work that would be done with single-rate
    /// integration to the actual work done with multi-rate.
    /// Values > 1.0 indicate efficiency gain.
    pub fn efficiency_ratio(&self) -> f64 {
        if self.subcycle_counts.is_empty() || self.total_steps == 0 {
            return 1.0;
        }

        // Compute actual work done (sum of all subcycles)
        let actual_work: usize = self.subcycle_counts.values().sum();

        // Compute work if single-rate was used (everyone at fastest rate)
        // This would be: total_steps * max_subcycles_per_component * n_components
        let max_subcycles = self.subcycle_counts.values().max().copied().unwrap_or(1);
        let single_rate_work = self.total_steps * max_subcycles * self.subcycle_counts.len();

        single_rate_work as f64 / actual_work.max(1) as f64
    }
}
