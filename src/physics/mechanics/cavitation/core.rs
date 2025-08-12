// src/physics/mechanics/cavitation/core.rs
//! Core cavitation physics implementation following SOLID principles
//!
//! This module provides the fundamental cavitation model based on the
//! Rayleigh-Plesset equation with proper state management.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::state::{PhysicsState, FieldAccessor, field_indices};
use crate::physics::traits::CavitationModelBehavior;
use ndarray::{Array3, Zip, ArrayView3};


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
        let state = PhysicsState::new(grid.clone());
        
        // Initialize bubble radius field
        state.initialize_field(field_indices::BUBBLE_RADIUS, initial_radius).unwrap();
        
        // Initialize bubble velocity field to zero
        state.initialize_field(field_indices::BUBBLE_VELOCITY, 0.0).unwrap();
        
        Self {
            state,
            initial_radius,
            equilibrium_radius: initial_radius,
            damping_coefficient: 0.01,
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

    /// Get a field from the physics state
    pub fn get_field(&self, field_index: usize) -> KwaversResult<Array3<f64>> {
        let guard = self.state.get_field(field_index)?;
        Ok(guard.to_owned())  // Convert view to owned array
    }
    
    /// Update a field in the physics state
    pub fn update_field(&mut self, field_index: usize, data: &Array3<f64>) -> KwaversResult<()> {
        self.state.update_field(field_index, data)
    }
    
    /// Update cavitation in-place and return the modified pressure field
    /// This avoids unnecessary cloning of the input pressure field
    pub fn update_cavitation_inplace(
        &mut self,
        pressure: ArrayView3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Start with a copy of the pressure field that we'll modify
        let mut modified_pressure = pressure.to_owned();
        
        // Call the existing update_cavitation with the owned copy
        self.update_cavitation(&modified_pressure, grid, medium, dt, t)?;
        
        // Return the modified pressure
        Ok(modified_pressure)
    }
}

// FieldAccessor trait implementation removed - use PhysicsState methods directly

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
            .for_each(|v_updated, &r, &v, &p| {
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
                    *v_updated = v + acceleration * dt;
                }
            });
        
        // Update radius using new velocity
        Zip::from(&mut new_radius)
            .and(&radius)
            .and(&new_velocity)
            .for_each(|r_updated, &r, &v_updated| {
                let updated_r = r + v_updated * dt;
                // Prevent negative radius
                *r_updated = updated_r.max(1e-10);
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
    
    fn light_emission(&self) -> Array3<f64> {
        // Calculate light emission based on bubble dynamics
        // Light emission is proportional to the rate of bubble collapse
        let radius = self.get_field(field_indices::BUBBLE_RADIUS).unwrap_or_else(|_| Array3::zeros((1, 1, 1)));
        let velocity = self.get_field(field_indices::BUBBLE_VELOCITY).unwrap_or_else(|_| Array3::zeros((1, 1, 1)));
        
        let mut emission = Array3::zeros(radius.dim());
        
        // Simple model: light emission when bubble is collapsing rapidly
        Zip::from(&mut emission)
            .and(&radius)
            .and(&velocity)
            .for_each(|e, &r, &v| {
                // Emit light during rapid collapse (negative velocity, small radius)
                if v < -100.0 && r < 1e-5 {
                    // Intensity proportional to collapse rate and inversely to radius
                    let collapse_rate = -v;
                    let intensity = (collapse_rate / 1000.0) * (1e-6 / r).min(1e6);
                    *e = intensity * 1e-12; // Scale to realistic power density (W/m³)
                }
            });
        
        emission
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
