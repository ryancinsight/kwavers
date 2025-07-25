//! Acoustic scattering physics module
//! 
//! Implements various acoustic scattering models following SOLID principles.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::state::{PhysicsState, FieldAccessor, field_indices};
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

pub mod rayleigh;
pub mod mie;
pub mod bubble_interactions;

pub use rayleigh::{RayleighScattering, compute_rayleigh_scattering};
pub use mie::compute_mie_scattering;
pub use bubble_interactions::compute_bubble_interactions;

/// Base acoustic scattering model
#[derive(Debug, Clone)]
pub struct AcousticScattering {
    /// Physics state container
    state: PhysicsState,
    
    /// Scattering parameters
    scattering_strength: f64,
    frequency: f64,
    
    /// Computed scattered field
    scattered_field: Array3<f64>,
    
    /// Performance tracking
    computation_time: std::time::Duration,
    update_count: usize,
}

impl AcousticScattering {
    /// Create a new acoustic scattering model
    pub fn new(grid: &Grid, frequency: f64, scattering_strength: f64) -> Self {
        let state = PhysicsState::new(grid);
        let scattered_field = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        Self {
            state,
            scattering_strength,
            frequency,
            scattered_field,
            computation_time: std::time::Duration::ZERO,
            update_count: 0,
        }
    }
    
    /// Compute scattering from particles/bubbles
    pub fn compute_scattering(
        &mut self,
        incident_field: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();
        
        // Get bubble properties from state
        let bubble_radius = self.get_field(field_indices::BUBBLE_RADIUS)?;
        let bubble_velocity = self.get_field(field_indices::BUBBLE_VELOCITY)?;
        
        // Reset scattered field
        self.scattered_field.fill(0.0);
        
        // Compute scattering based on bubble properties
        let k = 2.0 * PI * self.frequency / medium.sound_speed(0.0, 0.0, 0.0, grid);
        
        Zip::from(&mut self.scattered_field)
            .and(incident_field)
            .and(&bubble_radius)
            .and(&bubble_velocity)
            .for_each(|s, &p_inc, &r, &v| {
                if r > 0.0 {
                    // Simple monopole scattering model
                    let ka = k * r;
                    let scattering_cross_section = 4.0 * PI * r * r * (ka * ka) / (1.0 + ka * ka);
                    
                    // Include velocity effects (Doppler)
                    let doppler_factor = 1.0 + v / medium.sound_speed(0.0, 0.0, 0.0, grid);
                    
                    *s = self.scattering_strength * scattering_cross_section * p_inc * doppler_factor;
                }
            });
        
        self.computation_time += start_time.elapsed();
        self.update_count += 1;
        
        Ok(())
    }
    
    /// Get the computed scattered field
    pub fn scattered_field(&self) -> &Array3<f64> {
        &self.scattered_field
    }
    
    /// Report performance metrics
    pub fn report_performance(&self) {
        if self.update_count > 0 {
            let avg_time = self.computation_time.as_secs_f64() / self.update_count as f64;
            log::info!(
                "AcousticScattering Performance: {} updates, {:.3} ms average per update",
                self.update_count,
                avg_time * 1000.0
            );
        }
    }
}

impl FieldAccessor for AcousticScattering {
    fn physics_state(&self) -> &PhysicsState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HomogeneousMedium;
    
    #[test]
    fn test_acoustic_scattering() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let mut scattering = AcousticScattering::new(&grid, 1e6, 0.1);
        
        // Initialize bubble field
        scattering.state.initialize_field(field_indices::BUBBLE_RADIUS, 1e-6).unwrap();
        
        // Create incident field
        let incident = Array3::from_elem((10, 10, 10), 1000.0);
        
        // Create test medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
        
        // Compute scattering
        scattering.compute_scattering(&incident, &grid, &medium).unwrap();
        
        // Check that scattered field is non-zero
        let scattered = scattering.scattered_field();
        assert!(scattered.iter().any(|&x| x != 0.0));
    }
}
