//! Rayleigh scattering implementation for small particles
//!
//! Implements Rayleigh scattering for particles much smaller than the wavelength.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::state::{PhysicsState, FieldAccessor, field_indices};
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Rayleigh scattering model for small particles
#[derive(Debug, Clone)]
pub struct RayleighScattering {
    /// Physics state container
    state: PhysicsState,
    
    /// Model parameters
    frequency: f64,
    particle_density: f64,
    
    /// Computed scattered field
    scattered_field: Array3<f64>,
    
    /// Performance tracking
    computation_time: std::time::Duration,
    update_count: usize,
}

impl RayleighScattering {
    /// Create a new Rayleigh scattering model
    pub fn new(grid: &Grid, frequency: f64, particle_density: f64) -> Self {
        let state = PhysicsState::new(grid.clone());
        let scattered_field = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        Self {
            state,
            frequency,
            particle_density,
            scattered_field,
            computation_time: std::time::Duration::ZERO,
            update_count: 0,
        }
    }
    
    /// Compute Rayleigh scattering
    pub fn compute_scattering(
        &mut self,
        incident_field: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();
        
        // Get bubble radius from state
        let bubble_radius = self.state.get_field(field_indices::BUBBLE_RADIUS)?;
        
        // Medium properties
        let sound_speed = medium.sound_speed(0.0, 0.0, 0.0, grid);
        let density = medium.density(0.0, 0.0, 0.0, grid);
        
        // Wave properties
        let wavelength = sound_speed / self.frequency;
        let k = 2.0 * PI / wavelength;
        
        // Compute Rayleigh scattering
        Zip::from(&mut self.scattered_field)
            .and(incident_field)
            .and(bubble_radius)
            .apply(|s, &p_inc, &r| {
                if r > 0.0 && r < wavelength / 10.0 {  // Rayleigh regime
                    // Rayleigh scattering cross-section
                    let kr = k * r;
                    let kr2 = kr * kr;
                    let kr4 = kr2 * kr2;
                    
                    // Density contrast factor
                    let density_ratio = self.particle_density / density;
                    let contrast = (density_ratio - 1.0) / (density_ratio + 2.0);
                    
                    // Rayleigh scattering amplitude
                    let sigma_rayleigh = (8.0 * PI / 3.0) * r * r * kr4 * contrast * contrast;
                    
                    *s = sigma_rayleigh * p_inc;
                }
            });
        
        self.computation_time += start_time.elapsed();
        self.update_count += 1;
        
        Ok(())
    }
    
    /// Get the scattered field
    pub fn scattered_field(&self) -> &Array3<f64> {
        &self.scattered_field
    }
    
    /// Report performance metrics
    pub fn report_performance(&self) {
        if self.update_count > 0 {
            let avg_time = self.computation_time.as_secs_f64() / self.update_count as f64;
            log::info!(
                "RayleighScattering Performance: {} updates, {:.3} ms average per update",
                self.update_count,
                avg_time * 1000.0
            );
        }
    }
}

// FieldAccessor trait implementation removed - use PhysicsState methods directly

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HomogeneousMedium;
    
    #[test]
    fn test_rayleigh_scattering() {
        let grid = Grid::new(5, 5, 5, 0.1, 0.1, 0.1);
        let mut scattering = RayleighScattering::new(&grid, 1e6, 2000.0);
        
        // Initialize small bubbles (Rayleigh regime)
        scattering.state.initialize_field(field_indices::BUBBLE_RADIUS, 1e-7).unwrap();
        
        // Create incident field
        let incident = Array3::from_elem((5, 5, 5), 1000.0);
        
        // Create test medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
        
        // Compute scattering
        scattering.compute_scattering(&incident, &grid, &medium).unwrap();
        
        // Check that scattered field is computed
        let scattered = scattering.scattered_field();
        assert!(scattered.iter().any(|&x| x > 0.0));
    }
    
    #[test]
    fn test_rayleigh_regime_condition() {
        let grid = Grid::new(3, 3, 3, 0.1, 0.1, 0.1);
        let mut scattering = RayleighScattering::new(&grid, 1e6, 2000.0);
        
        // Initialize with large bubbles (outside Rayleigh regime)
        scattering.state.initialize_field(field_indices::BUBBLE_RADIUS, 1e-3).unwrap();
        
        let incident = Array3::from_elem((3, 3, 3), 1000.0);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
        
        scattering.compute_scattering(&incident, &grid, &medium).unwrap();
        
        // Should be zero for large bubbles
        let scattered = scattering.scattered_field();
        assert!(scattered.iter().all(|&x| x == 0.0));
    }
}

/// Compute Rayleigh scattering (backward compatibility function)
/// 
/// This function maintains backward compatibility with the old API.
pub fn compute_rayleigh_scattering(
    scatter: &mut Array3<f64>,
    radius: &Array3<f64>,
    p: &Array3<f64>,
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
) {
    // Create a temporary RayleighScattering model
    let mut model = RayleighScattering::new(grid, frequency, 2000.0); // Default particle density
    
    // Set the bubble radius field
    model.state.update_field(field_indices::BUBBLE_RADIUS, radius).unwrap_or_else(|e| {
        log::error!("Failed to update bubble radius: {}", e);
    });
    
    // Compute scattering
    if let Err(e) = model.compute_scattering(p, grid, medium) {
        log::error!("Failed to compute Rayleigh scattering: {}", e);
        scatter.fill(0.0);
        return;
    }
    
    // Copy result to output array
    scatter.assign(model.scattered_field());
}
