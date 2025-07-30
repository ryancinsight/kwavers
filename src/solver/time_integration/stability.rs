//! Stability analysis
//! 
//! This module provides stability analysis and CFL condition computation
//! for time integration methods.

use crate::grid::Grid;
use crate::KwaversResult;
use super::PhysicsComponent;
use ndarray::Array3;

/// Stability analyzer for time integration
#[derive(Debug)]
pub struct StabilityAnalyzer {
    /// Safety factor for CFL condition
    safety_factor: f64,
}

impl StabilityAnalyzer {
    /// Create a new stability analyzer
    pub fn new(safety_factor: f64) -> Self {
        Self { safety_factor }
    }
    
    /// Compute stable time step based on CFL condition
    pub fn compute_stable_dt(
        &self,
        physics: &dyn PhysicsComponent,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        // Get maximum wave speed
        let max_speed = physics.max_wave_speed(field, grid);
        
        // Compute CFL-limited time step
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_dt = self.safety_factor * dx_min / max_speed.max(1e-10);
        
        Ok(cfl_dt)
    }
    
    /// Check if a given time step is stable
    pub fn is_stable(
        &self,
        dt: f64,
        max_speed: f64,
        grid: &Grid,
    ) -> bool {
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_number = max_speed * dt / dx_min;
        
        cfl_number <= self.safety_factor
    }
    
    /// Compute CFL number
    pub fn compute_cfl_number(
        &self,
        dt: f64,
        max_speed: f64,
        grid: &Grid,
    ) -> f64 {
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        max_speed * dt / dx_min
    }
}

/// CFL condition information
#[derive(Debug, Clone)]
pub struct CFLCondition {
    /// Maximum stable time step
    pub max_dt: f64,
    /// Current CFL number
    pub cfl_number: f64,
    /// Maximum wave speed
    pub max_wave_speed: f64,
    /// Minimum grid spacing
    pub min_dx: f64,
    /// Is the current configuration stable?
    pub is_stable: bool,
}

impl CFLCondition {
    /// Create CFL condition from components
    pub fn new(
        dt: f64,
        max_wave_speed: f64,
        grid: &Grid,
        safety_factor: f64,
    ) -> Self {
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_number = max_wave_speed * dt / min_dx;
        let max_dt = safety_factor * min_dx / max_wave_speed.max(1e-10);
        let is_stable = cfl_number <= safety_factor;
        
        Self {
            max_dt,
            cfl_number,
            max_wave_speed,
            min_dx,
            is_stable,
        }
    }
    
    /// Get a stability report string
    pub fn report(&self) -> String {
        format!(
            "CFL Analysis: number={:.3}, max_dt={:.3e}, wave_speed={:.3e}, dx={:.3e}, stable={}",
            self.cfl_number, self.max_dt, self.max_wave_speed, self.min_dx, self.is_stable
        )
    }
}