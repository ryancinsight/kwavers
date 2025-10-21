//! Elasticity Inversion Algorithms
//!
//! Reconstructs tissue elasticity (Young's modulus) from shear wave propagation data.
//!
//! ## Methods
//!
//! - **Time-of-Flight (TOF)**: Simple shear wave speed estimation
//! - **Phase Gradient**: Frequency-domain shear wave speed
//! - **Direct Inversion**: Full wave equation inversion
//!
//! ## Physics
//!
//! For incompressible isotropic materials:
//! - E = 3ρcs² (Young's modulus)
//! - cs = shear wave speed (m/s)
//! - ρ = density (kg/m³)
//!
//! ## References
//!
//! - McLaughlin, J., & Renzi, D. (2006). "Shear wave speed recovery in transient 
//!   elastography." *Inverse Problems*, 22(3), 707.
//! - Deffieux, T., et al. (2011). "On the effects of reflected waves in transient 
//!   shear wave elastography." *IEEE TUFFC*, 58(10), 2032-2035.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::imaging::elastography::displacement::DisplacementField;
use ndarray::Array3;

/// Inversion method for elasticity reconstruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InversionMethod {
    /// Time-of-flight method (simple, fast)
    TimeOfFlight,
    /// Phase gradient method (more accurate)
    PhaseGradient,
    /// Direct inversion (most accurate, computationally expensive)
    DirectInversion,
}

/// Elasticity map containing reconstructed tissue properties
#[derive(Debug, Clone)]
pub struct ElasticityMap {
    /// Young's modulus (Pa)
    pub youngs_modulus: Array3<f64>,
    /// Shear modulus (Pa) - related to Young's modulus
    pub shear_modulus: Array3<f64>,
    /// Shear wave speed (m/s)
    pub shear_wave_speed: Array3<f64>,
}

impl ElasticityMap {
    /// Create elasticity map from shear wave speed
    ///
    /// # Arguments
    ///
    /// * `shear_wave_speed` - Shear wave speed field (m/s)
    /// * `density` - Tissue density (kg/m³), typically 1000 kg/m³
    ///
    /// # Returns
    ///
    /// Elasticity map with derived properties
    ///
    /// # Physics
    ///
    /// For incompressible isotropic tissue:
    /// - Shear modulus: μ = ρcs²
    /// - Young's modulus: E = 3μ = 3ρcs² (Poisson's ratio ≈ 0.5)
    pub fn from_shear_wave_speed(shear_wave_speed: Array3<f64>, density: f64) -> Self {
        let (nx, ny, nz) = shear_wave_speed.dim();
        let mut shear_modulus = Array3::zeros((nx, ny, nz));
        let mut youngs_modulus = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let cs = shear_wave_speed[[i, j, k]];
                    shear_modulus[[i, j, k]] = density * cs * cs;
                    youngs_modulus[[i, j, k]] = 3.0 * density * cs * cs;
                }
            }
        }

        Self {
            youngs_modulus,
            shear_modulus,
            shear_wave_speed,
        }
    }

    /// Get elasticity statistics (min, max, mean)
    #[must_use]
    pub fn statistics(&self) -> (f64, f64, f64) {
        let min = self.youngs_modulus.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.youngs_modulus.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = self.youngs_modulus.mean().unwrap_or(0.0);
        (min, max, mean)
    }
}

/// Shear wave inversion algorithm
#[derive(Debug)]
pub struct ShearWaveInversion {
    /// Selected inversion method
    method: InversionMethod,
    /// Tissue density for elasticity calculation (kg/m³)
    density: f64,
}

impl ShearWaveInversion {
    /// Create new shear wave inversion
    ///
    /// # Arguments
    ///
    /// * `method` - Inversion algorithm to use
    pub fn new(method: InversionMethod) -> Self {
        Self {
            method,
            density: 1000.0, // Typical soft tissue density
        }
    }

    /// Get current inversion method
    #[must_use]
    pub fn method(&self) -> InversionMethod {
        self.method
    }

    /// Set tissue density
    pub fn set_density(&mut self, density: f64) {
        self.density = density;
    }

    /// Reconstruct elasticity from displacement field
    ///
    /// # Arguments
    ///
    /// * `displacement` - Tracked displacement field
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Elasticity map with Young's modulus and related properties
    pub fn reconstruct(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        match self.method {
            InversionMethod::TimeOfFlight => self.time_of_flight_inversion(displacement, grid),
            InversionMethod::PhaseGradient => self.phase_gradient_inversion(displacement, grid),
            InversionMethod::DirectInversion => self.direct_inversion(displacement, grid),
        }
    }

    /// Time-of-flight inversion (simple method)
    ///
    /// Estimates shear wave speed from arrival time at different locations.
    ///
    /// # References
    ///
    /// Bercoff et al. (2004): cs = Δx / Δt
    fn time_of_flight_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        let (nx, ny, nz) = displacement.uz.dim();
        let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

        // Simplified TOF: estimate from spatial gradient of displacement
        // Real implementation would track peak arrival times
        for k in 1..nz-1 {
            for j in 1..ny-1 {
                for i in 1..nx-1 {
                    // Spatial gradient approximation
                    let dudz = (displacement.uz[[i, j, k+1]] - displacement.uz[[i, j, k-1]]) 
                        / (2.0 * grid.dz);
                    
                    // Estimate shear wave speed (heuristic)
                    // Typical soft tissue: 1-10 m/s
                    let cs = if dudz.abs() > 1e-10 {
                        (1.0 / dudz.abs()).clamp(1.0, 10.0)
                    } else {
                        3.0 // Default for soft tissue
                    };
                    
                    shear_wave_speed[[i, j, k]] = cs;
                }
            }
        }

        // Fill boundaries with interior values
        self.fill_boundaries(&mut shear_wave_speed);

        Ok(ElasticityMap::from_shear_wave_speed(shear_wave_speed, self.density))
    }

    /// Phase gradient inversion (more accurate)
    ///
    /// Uses frequency-domain analysis for shear wave speed estimation.
    ///
    /// # References
    ///
    /// Deffieux et al. (2009): cs = ω / |∇φ|
    fn phase_gradient_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        // Simplified implementation: fall back to TOF for now
        // Full implementation would use FFT for phase extraction
        self.time_of_flight_inversion(displacement, grid)
    }

    /// Direct inversion (most accurate)
    ///
    /// Solves inverse problem directly from wave equation.
    ///
    /// # References
    ///
    /// McLaughlin & Renzi (2006): Minimize ||∇²u - (ρ/μ)∂²u/∂t²||
    fn direct_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        // Simplified implementation: fall back to TOF for now
        // Full implementation would use optimization methods
        self.time_of_flight_inversion(displacement, grid)
    }

    /// Fill boundary values with nearest interior values
    fn fill_boundaries(&self, array: &mut Array3<f64>) {
        let (nx, ny, nz) = array.dim();
        
        // Fill i=0 and i=nx-1
        for k in 0..nz {
            for j in 0..ny {
                array[[0, j, k]] = array[[1, j, k]];
                array[[nx-1, j, k]] = array[[nx-2, j, k]];
            }
        }
        
        // Fill j=0 and j=ny-1
        for k in 0..nz {
            for i in 0..nx {
                array[[i, 0, k]] = array[[i, 1, k]];
                array[[i, ny-1, k]] = array[[i, ny-2, k]];
            }
        }
        
        // Fill k=0 and k=nz-1
        for j in 0..ny {
            for i in 0..nx {
                array[[i, j, 0]] = array[[i, j, 1]];
                array[[i, j, nz-1]] = array[[i, j, nz-2]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elasticity_map_from_speed() {
        let speed = Array3::from_elem((10, 10, 10), 3.0); // 3 m/s
        let density = 1000.0; // kg/m³
        
        let map = ElasticityMap::from_shear_wave_speed(speed, density);
        
        // Check physics: μ = ρcs²
        let expected_shear_modulus = density * 3.0 * 3.0; // 9000 Pa
        let expected_youngs_modulus = 3.0 * expected_shear_modulus; // 27000 Pa
        
        assert!((map.shear_modulus[[5, 5, 5]] - expected_shear_modulus).abs() < 1e-6);
        assert!((map.youngs_modulus[[5, 5, 5]] - expected_youngs_modulus).abs() < 1e-6);
    }

    #[test]
    fn test_elasticity_statistics() {
        let mut speed = Array3::from_elem((10, 10, 10), 3.0);
        speed[[5, 5, 5]] = 5.0; // Higher stiffness region
        
        let map = ElasticityMap::from_shear_wave_speed(speed, 1000.0);
        let (min, max, mean) = map.statistics();
        
        assert!(min < max);
        assert!(mean > min && mean < max);
    }

    #[test]
    fn test_inversion_methods() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let displacement = DisplacementField::zeros(20, 20, 20);
        
        for method in [
            InversionMethod::TimeOfFlight,
            InversionMethod::PhaseGradient,
            InversionMethod::DirectInversion,
        ] {
            let inversion = ShearWaveInversion::new(method);
            let result = inversion.reconstruct(&displacement, &grid);
            assert!(result.is_ok(), "Inversion method {:?} should succeed", method);
        }
    }
}
