//! Shear Wave Elastography (SWE) Module
//!
//! Implements clinical shear wave elastography for tissue characterization.
//!
//! ## Overview
//!
//! Shear wave elastography measures tissue stiffness by:
//! 1. Generating shear waves via acoustic radiation force impulse (ARFI)
//! 2. Tracking wave propagation with ultrafast imaging  
//! 3. Reconstructing elasticity from shear wave speed
//!
//! ## Literature References
//!
//! - Sarvazyan, A. P., et al. (1998). "Shear wave elasticity imaging: a new ultrasonic
//!   technology of medical diagnostics." *Ultrasound in Medicine & Biology*, 24(9), 1419-1435.
//! - Bercoff, J., et al. (2004). "Supersonic shear imaging: a new technique for soft tissue
//!   elasticity mapping." *IEEE TUFFC*, 51(4), 396-409.
//! - Deffieux, T., et al. (2009). "Shear wave spectroscopy for in vivo quantification of
//!   human soft tissues visco-elasticity." *IEEE TMI*, 28(3), 313-322.
//!
//! ## Clinical Applications
//!
//! - Liver fibrosis assessment (non-invasive)
//! - Breast tumor differentiation (benign vs malignant)
//! - Prostate cancer detection
//! - Thyroid nodule characterization

pub mod displacement;
pub mod inversion;
pub mod radiation_force;

pub use displacement::{DisplacementEstimator, DisplacementField};
pub use inversion::{ElasticityMap, InversionMethod, ShearWaveInversion};
pub use radiation_force::{AcousticRadiationForce, PushPulseParameters};

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;

/// Shear Wave Elastography configuration and workflow
///
/// # Example
///
/// ```no_run
/// use kwavers::physics::imaging::elastography::ShearWaveElastography;
/// use kwavers::physics::imaging::elastography::InversionMethod;
///
/// # fn example() -> kwavers::error::KwaversResult<()> {
/// let grid = kwavers::grid::Grid::new(100, 100, 100, 0.001, 0.001, 0.001)?;
/// let medium = kwavers::medium::HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
/// let swe = ShearWaveElastography::new(
///     &grid,
///     &medium,
///     InversionMethod::TimeOfFlight,
/// )?;
///
/// // Generate shear wave at focal point
/// let push_location = [0.05, 0.05, 0.05];
/// let displacement = swe.generate_shear_wave(push_location)?;
///
/// // Reconstruct elasticity map
/// let elasticity_map = swe.reconstruct_elasticity(&displacement)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ShearWaveElastography {
    /// Acoustic radiation force parameters
    radiation_force: AcousticRadiationForce,
    /// Displacement field estimator
    displacement_estimator: DisplacementEstimator,
    /// Elasticity inversion algorithm
    inversion: ShearWaveInversion,
    /// Computational grid
    grid: Grid,
}

impl ShearWaveElastography {
    /// Create new shear wave elastography workflow
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid for simulation
    /// * `medium` - Tissue medium properties
    /// * `inversion_method` - Algorithm for elasticity reconstruction
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        inversion_method: InversionMethod,
    ) -> KwaversResult<Self> {
        let radiation_force = AcousticRadiationForce::new(grid, medium)?;
        let displacement_estimator = DisplacementEstimator::new(grid);
        let inversion = ShearWaveInversion::new(inversion_method);

        Ok(Self {
            radiation_force,
            displacement_estimator,
            inversion,
            grid: grid.clone(),
        })
    }

    /// Generate shear wave using acoustic radiation force impulse (ARFI)
    ///
    /// # Arguments
    ///
    /// * `push_location` - Focal point for push pulse [x, y, z] in meters
    ///
    /// # Returns
    ///
    /// Displacement field as function of space (3D array: x, y, z)
    ///
    /// # References
    ///
    /// Sarvazyan et al. (1998): Acoustic radiation force generates shear waves
    /// perpendicular to ultrasound beam propagation direction.
    pub fn generate_shear_wave(&self, push_location: [f64; 3]) -> KwaversResult<Array3<f64>> {
        self.radiation_force.apply_push_pulse(push_location)
    }

    /// Reconstruct elasticity map from displacement field
    ///
    /// # Arguments
    ///
    /// * `displacement_field` - Tracked displacement as function of position
    ///
    /// # Returns
    ///
    /// Elasticity map with Young's modulus (Pa) at each grid point
    ///
    /// # References
    ///
    /// Bercoff et al. (2004): Shear wave speed cs relates to Young's modulus E
    /// via E = 3ρcs² for incompressible isotropic materials.
    pub fn reconstruct_elasticity(
        &self,
        displacement_field: &Array3<f64>,
    ) -> KwaversResult<ElasticityMap> {
        // Estimate displacement from tracked data
        let tracked_displacement = self.displacement_estimator.estimate(displacement_field)?;

        // Reconstruct elasticity using selected inversion method
        self.inversion
            .reconstruct(&tracked_displacement, &self.grid)
    }

    /// Get current push pulse parameters
    #[must_use]
    pub fn push_parameters(&self) -> &PushPulseParameters {
        self.radiation_force.parameters()
    }

    /// Get current inversion method
    #[must_use]
    pub fn inversion_method(&self) -> InversionMethod {
        self.inversion.method()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn test_swe_creation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let result = ShearWaveElastography::new(&grid, &medium, InversionMethod::TimeOfFlight);

        assert!(result.is_ok(), "SWE creation should succeed");
    }

    #[test]
    fn test_shear_wave_generation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let swe =
            ShearWaveElastography::new(&grid, &medium, InversionMethod::TimeOfFlight).unwrap();

        let push_location = [0.025, 0.025, 0.025]; // Center of grid
        let result = swe.generate_shear_wave(push_location);

        assert!(result.is_ok(), "Shear wave generation should succeed");

        let displacement = result.unwrap();
        let (nx, ny, nz) = displacement.dim();
        assert_eq!(nx, 50);
        assert_eq!(ny, 50);
        assert_eq!(nz, 50);
    }

    #[test]
    fn test_swe_elasticity_reconstruction() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let swe =
            ShearWaveElastography::new(&grid, &medium, InversionMethod::TimeOfFlight).unwrap();

        // Generate shear wave
        let push_location = [0.016, 0.016, 0.016]; // Near center
        let _displacement_field = swe.generate_shear_wave(push_location).unwrap();

        // Create synthetic displacement field for testing
        use displacement::DisplacementField;
        let mut synthetic_displacement = DisplacementField::zeros(grid.nx, grid.ny, grid.nz);

        // Add some synthetic displacement pattern
        let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r = (((i as i32 - center.0 as i32).pow(2) +
                             (j as i32 - center.1 as i32).pow(2) +
                             (k as i32 - center.2 as i32).pow(2)) as f64).sqrt();
                    let displacement = 1e-6 * (-r * 0.1).exp(); // Micro-meter scale
                    synthetic_displacement.ux[[i, j, k]] = displacement;
                    synthetic_displacement.uy[[i, j, k]] = displacement * 0.8;
                    synthetic_displacement.uz[[i, j, k]] = displacement * 0.6;
                }
            }
        }

        // Test elasticity reconstruction
        let elasticity_map = swe.reconstruct_elasticity(&synthetic_displacement.magnitude()).unwrap();

        // Validate reconstruction results
        let (nx, ny, nz) = elasticity_map.youngs_modulus.dim();
        assert_eq!(nx, grid.nx);
        assert_eq!(ny, grid.ny);
        assert_eq!(nz, grid.nz);

        // Check that we have reasonable elasticity values (should be positive)
        let min_elasticity = elasticity_map.youngs_modulus.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_elasticity = elasticity_map.youngs_modulus.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        assert!(min_elasticity > 0.0, "Elasticity should be positive");
        assert!(max_elasticity > min_elasticity, "Should have range of elasticity values");

        // Shear wave speed should also be reasonable
        let min_c = elasticity_map.shear_wave_speed.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_c = elasticity_map.shear_wave_speed.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        assert!(min_c > 0.0, "Shear wave speed should be positive");
        assert!(max_c > 0.0, "Shear wave speed should be positive");
    }
}
