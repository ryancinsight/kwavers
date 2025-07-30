//! Multi-rate controller
//! 
//! This module manages different time scales for multi-physics simulations.

use crate::KwaversResult;
use crate::error::{KwaversError, ValidationError};
use super::traits::MultiRateConfig;
use std::collections::HashMap;

/// Controller for multi-rate time integration
#[derive(Debug)]
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
        
        // Find minimum time step (global time step)
        let global_dt = component_time_steps
            .values()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .min(max_dt)
            .max(self.config.min_dt);
        
        // Compute subcycles for each component
        let mut subcycles = HashMap::new();
        
        for (name, &component_dt) in component_time_steps {
            let n_subcycles = ((component_dt / global_dt).floor() as usize)
                .max(1)
                .min(self.config.max_subcycles);
            
            subcycles.insert(name.clone(), n_subcycles);
            
            // Update statistics
            *self.subcycle_counts.entry(name.clone()).or_insert(0) += n_subcycles;
        }
        
        self.total_steps += 1;
        
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
    pub fn efficiency_ratio(&self) -> f64 {
        if self.subcycle_counts.is_empty() || self.total_steps == 0 {
            return 1.0;
        }
        
        // Compute actual work done
        let actual_work: usize = self.subcycle_counts.values().sum();
        
        // Compute work if single-rate was used
        let single_rate_work = self.total_steps * self.subcycle_counts.len();
        
        single_rate_work as f64 / actual_work.max(1) as f64
    }
}