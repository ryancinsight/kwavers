//! Chemical parameters and metrics
//!
//! This module contains parameter structures and metrics for chemical simulations
//! following the Single Responsibility Principle.

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;

/// Parameters for chemical update operations
/// Groups related parameters following SOLID principles
#[derive(Debug, Clone)]
pub struct ChemicalUpdateParams<'a> {
    pub pressure: &'a Array3<f64>,
    pub light: &'a Array3<f64>,
    pub emission_spectrum: &'a Array3<f64>,
    pub bubble_radius: &'a Array3<f64>,
    pub temperature: &'a Array3<f64>,
    pub grid: &'a Grid,
    pub dt: f64,
    pub medium: &'a dyn Medium,
    pub frequency: f64,
}

impl<'a> ChemicalUpdateParams<'a> {
    /// Create new chemical update parameters with validation
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pressure: &'a Array3<f64>,
        light: &'a Array3<f64>,
        emission_spectrum: &'a Array3<f64>,
        bubble_radius: &'a Array3<f64>,
        temperature: &'a Array3<f64>,
        grid: &'a Grid,
        dt: f64,
        medium: &'a dyn Medium,
        frequency: f64,
    ) -> KwaversResult<Self> {
        // Validate parameters
        if dt <= 0.0 {
            return Err(crate::error::NumericalError::InvalidOperation(
                "Time step must be positive".to_string(),
            )
            .into());
        }

        if frequency <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "frequency".to_string(),
                value: frequency,
                reason: "Frequency must be positive".to_string(),
            }
            .into());
        }

        // Validate array dimensions match grid
        let grid_shape = (grid.nx, grid.ny, grid.nz);
        let pressure_shape = pressure.dim();

        if pressure_shape != grid_shape {
            return Err(crate::KwaversError::InvalidInput(format!(
                "Pressure array shape {:?} doesn't match grid {:?}",
                pressure_shape, grid_shape
            )));
        }

        Ok(Self {
            pressure,
            light,
            emission_spectrum,
            bubble_radius,
            temperature,
            grid,
            dt,
            medium,
            frequency,
        })
    }

    /// Get time step
    pub fn time_step(&self) -> f64 {
        self.dt
    }

    /// Get driving frequency
    pub fn driving_frequency(&self) -> f64 {
        self.frequency
    }
}

/// Metrics for chemical model performance
#[derive(Debug, Clone, Default]
pub struct ChemicalMetrics {
    pub total_reactions: usize,
    pub computation_time_ms: f64,
    pub max_concentration_change: f64,
    pub average_reaction_rate: f64,
    pub stability_ratio: f64,
}

impl ChemicalMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Update computation time
    pub fn set_computation_time(&mut self, time_ms: f64) {
        self.computation_time_ms = time_ms;
    }

    /// Update reaction count
    pub fn increment_reactions(&mut self, count: usize) {
        self.total_reactions += count;
    }

    /// Update maximum concentration change
    pub fn update_max_change(&mut self, change: f64) {
        if change > self.max_concentration_change {
            self.max_concentration_change = change;
        }
    }

    /// Calculate and update average reaction rate
    pub fn update_average_rate(&mut self, rates: &[f64]) {
        if !rates.is_empty() {
            self.average_reaction_rate = rates.iter().sum::<f64>() / rates.len() as f64;
        }
    }

    /// Calculate stability ratio (should be < 1 for stability)
    pub fn calculate_stability(&mut self, dt: f64, max_rate: f64) {
        self.stability_ratio = dt * max_rate;
    }

    /// Check if the system is stable
    pub fn is_stable(&self) -> bool {
        self.stability_ratio < 1.0
    }
}
