// src/physics/mechanics/cavitation/core.rs
//! Core cavitation physics implementation following SOLID principles
//!
//! This module provides the fundamental cavitation model based on the
//! Rayleigh-Plesset equation with proper state management.

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::state::{PhysicsState, FieldAccessor, field_indices};
use crate::physics::traits::CavitationModelBehavior;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Core cavitation model implementing the Rayleigh-Plesset equation
#[derive(Debug, Clone)]
pub struct CavitationModel {
    /// Physics state container
    state: PhysicsState,
    
    /// Model parameters
    initial_radius: f64,
    equilibrium_radius: f64,
    damping_coefficient: f64,
    
    /// Performance tracking
    computation_time: std::time::Duration,
    update_count: usize,
}

impl CavitationModel {
    /// Create a new cavitation model
    pub fn new(grid: &Grid, initial_radius: f64) -> Self {
        let state = PhysicsState::new(grid);
        
        // Initialize bubble radius field
        state.initialize_field(field_indices::BUBBLE_RADIUS, initial_radius).unwrap();
        state.initialize_field(field_indices::BUBBLE_VELOCITY, 0.0).unwrap();
        
        Self {
            state,
            initial_radius,
            equilibrium_radius: initial_radius,
            damping_coefficient: 0.0,
            computation_time: std::time::Duration::ZERO,
            update_count: 0,
        }
    }
    
    /// Set the equilibrium bubble radius
    pub fn set_equilibrium_radius(&mut self, radius: f64) {
        self.equilibrium_radius = radius;
    }
    
    /// Set the damping coefficient
    pub fn set_damping_coefficient(&mut self, damping: f64) {
        self.damping_coefficient = damping;
    }
}

impl FieldAccessor for CavitationModel {
    fn physics_state(&self) -> &PhysicsState {
        &self.state
    }
}

impl CavitationModelBehavior for CavitationModel {
    fn update_cavitation(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();
        
        // Get current bubble state
        let radius = self.get_field(field_indices::BUBBLE_RADIUS)?;
        let velocity = self.get_field(field_indices::BUBBLE_VELOCITY)?;
        
        // Create new arrays for updated values
        let mut new_radius = radius.clone();
        let mut new_velocity = velocity.clone();
        
        // Update bubble dynamics using Rayleigh-Plesset equation
        Zip::from(&mut new_velocity)
            .and(&radius)
            .and(&velocity)
            .and(pressure)
            .for_each(|v_new, &r, &v, &p| {
                if r > 0.0 {
                    // Get medium properties at bubble location
                    let rho = medium.density(0.0, 0.0, 0.0, grid);
                    let sigma = medium.surface_tension(0.0, 0.0, 0.0, grid);
                    let mu = medium.viscosity(0.0, 0.0, 0.0, grid);
                    let p_ambient = medium.ambient_pressure(0.0, 0.0, 0.0, grid);
                    let p_vapor = medium.vapor_pressure(0.0, 0.0, 0.0, grid);
                    
                    // Pressure difference driving bubble dynamics
                    let p_gas = (p_ambient + 2.0 * sigma / self.initial_radius) 
                        * (self.initial_radius / r).powi(3);
                    let p_diff = p_gas - p - p_vapor + 2.0 * sigma / r;
                    
                    // Rayleigh-Plesset acceleration
                    let viscous_term = 4.0 * mu * v / r;
                    let acceleration = (p_diff - viscous_term) / (rho * r) 
                        - 1.5 * v * v / r
                        - self.damping_coefficient * v;
                    
                    // Update velocity (semi-implicit for stability)
                    *v_new = v + acceleration * dt;
                }
            });
        
        // Update radius using new velocity
        Zip::from(&mut new_radius)
            .and(&radius)
            .and(&new_velocity)
            .for_each(|r_new, &r, &v_new| {
                let new_r = r + v_new * dt;
                // Prevent negative radius
                *r_new = new_r.max(1e-10);
            });
        
        // Update state
        self.update_field(field_indices::BUBBLE_RADIUS, &new_radius)?;
        self.update_field(field_indices::BUBBLE_VELOCITY, &new_velocity)?;
        
        self.computation_time += start_time.elapsed();
        self.update_count += 1;
        
        Ok(())
    }
    
    fn bubble_radius(&self) -> KwaversResult<Array3<f64>> {
        self.get_field(field_indices::BUBBLE_RADIUS)
    }
    
    fn bubble_velocity(&self) -> KwaversResult<Array3<f64>> {
        self.get_field(field_indices::BUBBLE_VELOCITY)
    }
    
    fn report_performance(&self) {
        if self.update_count > 0 {
            let avg_time = self.computation_time.as_secs_f64() / self.update_count as f64;
            log::info!(
                "CavitationModel Performance: {} updates, {:.3} ms average per update",
                self.update_count,
                avg_time * 1000.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HomogeneousMedium;
    
    #[test]
    fn test_cavitation_model() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let mut model = CavitationModel::new(&grid, 1e-6);
        
        // Create test pressure field
        let pressure = Array3::from_elem((10, 10, 10), 101325.0);
        
        // Create test medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
        
        // Update cavitation
        model.update_cavitation(&pressure, &grid, &medium, 1e-6, 0.0).unwrap();
        
        // Check that bubble radius was updated
        let radius = model.bubble_radius().unwrap();
        assert!(radius[[5, 5, 5]] > 0.0);
    }
    
    #[test]
    fn test_rayleigh_plesset_dynamics() {
        let grid = Grid::new(5, 5, 5, 0.1, 0.1, 0.1);
        let mut model = CavitationModel::new(&grid, 1e-6);
        model.set_damping_coefficient(0.1);
        
        // Create varying pressure field
        let mut pressure = Array3::zeros((5, 5, 5));
        pressure[[2, 2, 2]] = 50000.0; // Low pressure to induce growth
        
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
        
        // Run multiple time steps
        let initial_radius = model.bubble_radius().unwrap()[[2, 2, 2]];
        for _ in 0..10 {
            model.update_cavitation(&pressure, &grid, &medium, 1e-7, 0.0).unwrap();
        }
        let final_radius = model.bubble_radius().unwrap()[[2, 2, 2]];
        
        // Bubble should grow under low pressure
        assert!(final_radius > initial_radius);
    }
}
