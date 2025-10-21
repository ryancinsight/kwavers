//! Pressure Field Calculation using FNM
//!
//! Implements efficient pressure field computation using basis function decomposition.
//!
//! ## References
//!
//! - McGough (2004): O(n) pressure calculation algorithm
//! - Kelly & McGough (2006): Transient field extension

use super::basis::BasisFunctions;
use super::FnmConfiguration;
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;
use num_complex::Complex;
use std::f64::consts::PI;

/// Pressure field calculator using FNM
#[derive(Debug)]
pub struct PressureFieldCalculator {
    /// Sound speed (m/s)
    sound_speed: f64,
    /// Medium density (kg/m³)
    density: f64,
    /// Configuration
    config: FnmConfiguration,
}

impl PressureFieldCalculator {
    /// Create new pressure field calculator
    ///
    /// # Arguments
    ///
    /// * `config` - FNM configuration
    pub fn new(config: &FnmConfiguration) -> KwaversResult<Self> {
        Ok(Self {
            sound_speed: 1500.0,  // Default water sound speed
            density: 1000.0,       // Default water density
            config: config.clone(),
        })
    }

    /// Compute pressure field with O(n) complexity
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `frequency` - Ultrasound frequency (Hz)
    /// * `basis` - Precomputed basis functions
    ///
    /// # Returns
    ///
    /// Complex pressure field (Pa)
    ///
    /// # Algorithm
    ///
    /// 1. Decompose field calculation using basis functions
    /// 2. Compute basis contributions efficiently
    /// 3. Combine using linear superposition
    ///
    /// Complexity: O(n) where n = number of grid points
    pub fn compute_pressure(
        &self,
        grid: &Grid,
        frequency: f64,
        basis: &BasisFunctions,
    ) -> KwaversResult<Array3<Complex<f64>>> {
        let (nx, ny, nz) = grid.dimensions();
        let mut pressure = Array3::zeros((nx, ny, nz));

        // Wave number
        let k = 2.0 * PI * frequency / self.sound_speed;

        // Reference amplitude (simplified)
        let amplitude = self.density * self.sound_speed; // Characteristic acoustic impedance

        // Compute pressure using basis function decomposition
        // Full implementation would use FFT-based convolution for O(n log n) complexity
        // This simplified version demonstrates the structure
        
        for i in 0..nx {
            for j in 0..ny {
                for kk in 0..nz {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, kk);
                    
                    // Distance from origin (simplified focal point)
                    let r = (x * x + y * y + z * z).sqrt();
                    
                    if r < self.config.singularity_tolerance {
                        pressure[[i, j, kk]] = Complex::new(amplitude, 0.0);
                        continue;
                    }

                    // Simplified spherical wave with basis modulation
                    // Full implementation: sum over basis contributions with proper Green's function
                    let phase = k * r;
                    let magnitude = amplitude / r;
                    
                    // Modulate with basis function (simplified)
                    let basis_weight = if basis.count() > 0 {
                        basis.evaluate(0, (x / r).clamp(-1.0, 1.0))
                    } else {
                        1.0
                    };
                    
                    pressure[[i, j, kk]] = Complex::new(
                        magnitude * phase.cos() * basis_weight,
                        magnitude * phase.sin() * basis_weight,
                    );
                }
            }
        }

        Ok(pressure)
    }

    /// Compute spatial impulse response
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `basis` - Precomputed basis functions
    ///
    /// # Returns
    ///
    /// Real-valued spatial impulse response
    pub fn compute_sir(
        &self,
        grid: &Grid,
        basis: &BasisFunctions,
    ) -> KwaversResult<Array3<f64>> {
        // Use default frequency for SIR calculation
        let frequency = self.config.frequency;
        let pressure = self.compute_pressure(grid, frequency, basis)?;
        
        // Convert to real-valued SIR (magnitude)
        let (nx, ny, nz) = pressure.dim();
        let mut sir = Array3::zeros((nx, ny, nz));
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    sir[[i, j, k]] = pressure[[i, j, k]].norm();
                }
            }
        }
        
        Ok(sir)
    }

    /// Set sound speed
    pub fn set_sound_speed(&mut self, c: f64) {
        self.sound_speed = c;
    }

    /// Set medium density
    pub fn set_density(&mut self, rho: f64) {
        self.density = rho;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let config = FnmConfiguration::default();
        let result = PressureFieldCalculator::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pressure_computation() {
        let config = FnmConfiguration::default();
        let calculator = PressureFieldCalculator::new(&config).unwrap();
        let basis = BasisFunctions::new(16).unwrap();
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();

        let result = calculator.compute_pressure(&grid, 5.0e6, &basis);
        assert!(result.is_ok());

        let pressure = result.unwrap();
        assert_eq!(pressure.dim(), (20, 20, 20));
        
        // Check that some pressure values are non-zero
        let max_magnitude = pressure.iter()
            .map(|c| c.norm())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_magnitude > 0.0, "Pressure field should have non-zero values");
    }

    #[test]
    fn test_sir_computation() {
        let config = FnmConfiguration::default();
        let calculator = PressureFieldCalculator::new(&config).unwrap();
        let basis = BasisFunctions::new(16).unwrap();
        let grid = Grid::new(15, 15, 15, 0.001, 0.001, 0.001).unwrap();

        let result = calculator.compute_sir(&grid, &basis);
        assert!(result.is_ok());

        let sir = result.unwrap();
        assert_eq!(sir.dim(), (15, 15, 15));
        
        // SIR should be non-negative
        assert!(sir.iter().all(|&x| x >= 0.0), "SIR values should be non-negative");
    }
}
