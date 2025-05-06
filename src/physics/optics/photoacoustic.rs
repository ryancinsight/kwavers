use crate::grid::Grid;
use crate::medium::Medium;
use crate::source::Source;
use ndarray::Array3;
use std::sync::Arc;

#[derive(Debug)]
pub struct PhotoacousticSource {
    initial_pressure: Array3<f64>,
    medium: Arc<dyn Medium>,
    grüneisen_parameter: f64,  // Grüneisen parameter for photoacoustic efficiency
}

impl PhotoacousticSource {
    /// Create a new photoacoustic source with default Grüneisen parameter
    pub fn new(initial_pressure: Array3<f64>) -> Self {
        let grid = Grid::new(
            initial_pressure.shape()[0],
            initial_pressure.shape()[1],
            initial_pressure.shape()[2],
            1.0, 1.0, 1.0  // Default grid spacing, will be overridden in source term calculation
        );
        
        // Create a default medium (water-like)
        let medium = Arc::new(crate::medium::homogeneous::HomogeneousMedium::new(
            1000.0,  // density (kg/m³)
            1500.0,  // sound speed (m/s)
            &grid,
            0.0,     // absorption coefficient
            0.0,     // scattering coefficient
        ));
        
        Self::new_with_medium(initial_pressure, medium)
    }
    
    /// Create a new photoacoustic source with specific medium properties
    pub fn new_with_medium(initial_pressure: Array3<f64>, medium: Arc<dyn Medium>) -> Self {
        PhotoacousticSource {
            initial_pressure,
            medium,
            grüneisen_parameter: 0.12,  // Default value for water/tissue
        }
    }
    
    /// Set the Grüneisen parameter (dimensionless photoacoustic efficiency)
    pub fn set_gruneisen_parameter(&mut self, value: f64) {
        self.grüneisen_parameter = value;
    }
    
    /// Calculate photoacoustic pressure based on optical absorption
    fn calculate_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Get grid indices
        let i = grid.x_idx(x);
        let j = grid.y_idx(y);
        let k = grid.z_idx(z);
        
        if i >= self.initial_pressure.shape()[0] || 
           j >= self.initial_pressure.shape()[1] || 
           k >= self.initial_pressure.shape()[2] {
            return 0.0;
        }
        
        // Get local tissue properties
        let absorption = self.medium.absorption_coefficient_light(x, y, z, grid);
        let thermal_expansion = self.medium.thermal_expansion(x, y, z, grid);
        let specific_heat = self.medium.specific_heat(x, y, z, grid);
        let sound_speed = self.medium.sound_speed(x, y, z, grid);
        
        // Calculate local Grüneisen parameter based on tissue properties
        let local_gruneisen = thermal_expansion * sound_speed * sound_speed / specific_heat;
        
        // Use either tissue-specific or default Grüneisen parameter
        let gamma = if local_gruneisen > 0.0 { local_gruneisen } else { self.grüneisen_parameter };
        
        // Get initial pressure at this point
        let p0 = self.initial_pressure[[i, j, k]];
        
        // Apply photoacoustic effect
        gamma * absorption * p0
    }
}

impl Source for PhotoacousticSource {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if t == 0.0 {
            // Initial pressure distribution
            self.calculate_pressure(x, y, z, grid)
        } else {
            // No additional source terms after initial condition
            0.0
        }
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Return positions of non-zero initial pressure
        let mut positions = Vec::new();
        let shape = self.initial_pressure.shape();
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    if self.initial_pressure[[i, j, k]] > 0.0 {
                        positions.push((i as f64, j as f64, k as f64));
                    }
                }
            }
        }
        
        positions
    }
    
    fn signal(&self) -> &dyn crate::signal::Signal {
        // Photoacoustic sources don't have a time-varying signal
        // They use the initial pressure distribution instead
        panic!("Photoacoustic sources do not provide a time-varying signal");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_photoacoustic_source_creation() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let mut initial_pressure = Array3::<f64>::zeros((nx, ny, nz));
        initial_pressure[[5, 5, 5]] = 1.0;
        
        let source = PhotoacousticSource::new(initial_pressure);
        assert_relative_eq!(source.grüneisen_parameter, 0.12);
    }
    
    #[test]
    fn test_photoacoustic_pressure_calculation() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let grid = Grid::new(nx, ny, nz, 0.001, 0.001, 0.001);
        
        let mut initial_pressure = Array3::<f64>::zeros((nx, ny, nz));
        initial_pressure[[5, 5, 5]] = 1.0;
        
        let source = PhotoacousticSource::new(initial_pressure);
        
        // Test pressure at source point
        let p = source.get_source_term(0.0, 0.005, 0.005, 0.005, &grid);
        assert!(p > 0.0);
        
        // Test pressure away from source
        let p = source.get_source_term(0.0, 0.009, 0.009, 0.009, &grid);
        assert_relative_eq!(p, 0.0);
        
        // Test pressure at later time
        let p = source.get_source_term(1e-6, 0.005, 0.005, 0.005, &grid);
        assert_relative_eq!(p, 0.0);
    }
}
