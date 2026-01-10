//! Axisymmetric solver configuration
//!
//! Configuration structures for the axisymmetric k-space pseudospectral solver.

#![allow(deprecated)]

use serde::{Deserialize, Serialize};

/// Configuration for axisymmetric solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisymmetricConfig {
    /// Number of grid points in axial (z) direction
    pub nz: usize,
    /// Number of grid points in radial (r) direction
    pub nr: usize,
    /// Grid spacing in axial direction (m)
    pub dz: f64,
    /// Grid spacing in radial direction (m)
    pub dr: f64,
    /// Time step (s)
    pub dt: f64,
    /// Number of time steps
    pub nt: usize,
    /// PML thickness in grid points
    pub pml_size: usize,
    /// PML absorption coefficient
    pub pml_alpha: f64,
    /// Use k-space correction for improved accuracy
    pub use_kspace_correction: bool,
    /// Reference sound speed for k-space correction (m/s)
    pub c_ref: f64,
    /// Enable nonlinear propagation
    pub nonlinear: bool,
    /// B/A nonlinearity parameter (if nonlinear is true)
    pub b_over_a: f64,
}

impl Default for AxisymmetricConfig {
    fn default() -> Self {
        Self {
            nz: 128,
            nr: 64,
            dz: 1e-4, // 100 μm
            dr: 1e-4, // 100 μm
            dt: 1e-8, // 10 ns
            nt: 1000,
            pml_size: 20,
            pml_alpha: 2.0,
            use_kspace_correction: true,
            c_ref: 1500.0, // Water
            nonlinear: false,
            b_over_a: 5.0,
        }
    }
}

impl AxisymmetricConfig {
    /// Create configuration for a typical HIFU simulation
    pub fn hifu_default() -> Self {
        Self {
            nz: 256,
            nr: 128,
            dz: 0.5e-3, // 0.5 mm
            dr: 0.5e-3, // 0.5 mm
            dt: 2e-8,   // 20 ns
            nt: 5000,
            pml_size: 30,
            pml_alpha: 2.0,
            use_kspace_correction: true,
            c_ref: 1540.0, // Tissue
            nonlinear: true,
            b_over_a: 6.0,
        }
    }

    /// Create configuration for photoacoustic imaging
    pub fn photoacoustic_default() -> Self {
        Self {
            nz: 128,
            nr: 128,
            dz: 50e-6, // 50 μm
            dr: 50e-6, // 50 μm
            dt: 5e-9,  // 5 ns
            nt: 2000,
            pml_size: 20,
            pml_alpha: 2.0,
            use_kspace_correction: true,
            c_ref: 1500.0,
            nonlinear: false,
            b_over_a: 5.0,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.nz == 0 || self.nr == 0 {
            return Err("Grid dimensions must be positive".to_string());
        }
        if self.dz <= 0.0 || self.dr <= 0.0 {
            return Err("Grid spacing must be positive".to_string());
        }
        if self.dt <= 0.0 {
            return Err("Time step must be positive".to_string());
        }
        if self.c_ref <= 0.0 {
            return Err("Reference sound speed must be positive".to_string());
        }
        if self.pml_size >= self.nz.min(self.nr) / 2 {
            return Err("PML size too large for grid dimensions".to_string());
        }
        Ok(())
    }

    /// Calculate CFL number for stability check
    pub fn cfl_number(&self, c_max: f64) -> f64 {
        // For axisymmetric: CFL = c_max * dt / min(dr, dz)
        c_max * self.dt / self.dr.min(self.dz)
    }

    /// Check if CFL condition is satisfied
    pub fn is_stable(&self, c_max: f64) -> bool {
        // For k-space methods, CFL < 0.5 is typically safe
        self.cfl_number(c_max) < 0.5
    }
}

/// Medium properties for axisymmetric simulation
///
/// # Deprecation
///
/// This type is deprecated in favor of using domain-level `Medium` types with
/// `CylindricalMediumProjection`. The new approach provides better separation
/// of concerns and allows reuse of domain medium definitions.
///
/// # Migration
///
/// Replace direct use of `AxisymmetricMedium` with:
/// 1. Create a 3D `Medium` (e.g., `HomogeneousMedium`, `HeterogeneousMedium`)
/// 2. Create a `CylindricalMediumProjection` to project it to 2D
/// 3. Use `AxisymmetricSolver::new_with_projection` instead of `new`
///
/// See [`AxisymmetricSolver::new_with_projection`] for examples.
///
/// [`AxisymmetricSolver::new_with_projection`]: crate::solver::forward::axisymmetric::AxisymmetricSolver::new_with_projection
#[deprecated(
    since = "2.16.0",
    note = "Use domain-level `Medium` types with `CylindricalMediumProjection` instead. \
            See type documentation for migration guide."
)]
#[derive(Debug, Clone)]
pub struct AxisymmetricMedium {
    /// Sound speed field (nz x nr)
    pub sound_speed: ndarray::Array2<f64>,
    /// Density field (nz x nr)
    pub density: ndarray::Array2<f64>,
    /// Absorption coefficient (Np/m) at reference frequency
    pub alpha_coeff: ndarray::Array2<f64>,
    /// Absorption power law exponent
    pub alpha_power: f64,
    /// B/A nonlinearity parameter (optional)
    pub b_over_a: Option<ndarray::Array2<f64>>,
}

impl AxisymmetricMedium {
    /// Create homogeneous medium
    ///
    /// # Deprecation
    ///
    /// This method is deprecated. Use `HomogeneousMedium` from `domain::medium` instead.
    #[deprecated(
        since = "2.16.0",
        note = "Use `HomogeneousMedium::new` with `CylindricalMediumProjection` instead"
    )]
    pub fn homogeneous(nz: usize, nr: usize, sound_speed: f64, density: f64) -> Self {
        Self {
            sound_speed: ndarray::Array2::from_elem((nz, nr), sound_speed),
            density: ndarray::Array2::from_elem((nz, nr), density),
            alpha_coeff: ndarray::Array2::zeros((nz, nr)),
            alpha_power: 2.0, // Water-like
            b_over_a: None,
        }
    }

    /// Create tissue medium with typical properties
    ///
    /// # Deprecation
    ///
    /// This method is deprecated. Use `HomogeneousMedium` from `domain::medium` instead.
    #[deprecated(
        since = "2.16.0",
        note = "Use `HomogeneousMedium::new` with tissue parameters instead"
    )]
    pub fn tissue(nz: usize, nr: usize) -> Self {
        Self {
            sound_speed: ndarray::Array2::from_elem((nz, nr), 1540.0),
            density: ndarray::Array2::from_elem((nz, nr), 1050.0),
            alpha_coeff: ndarray::Array2::from_elem((nz, nr), 0.5), // ~0.5 Np/m/MHz
            alpha_power: 1.1,                                       // Typical tissue
            b_over_a: Some(ndarray::Array2::from_elem((nz, nr), 6.0)),
        }
    }

    /// Get maximum sound speed for CFL calculation
    ///
    /// # Deprecation
    ///
    /// This method is deprecated. Use `CylindricalMediumProjection::max_sound_speed` instead.
    #[deprecated(
        since = "2.16.0",
        note = "Use `CylindricalMediumProjection::max_sound_speed` instead"
    )]
    pub fn max_sound_speed(&self) -> f64 {
        self.sound_speed.iter().cloned().fold(f64::MIN, f64::max)
    }

    /// Get minimum sound speed
    ///
    /// # Deprecation
    ///
    /// This method is deprecated. Use `CylindricalMediumProjection::min_sound_speed` instead.
    #[deprecated(
        since = "2.16.0",
        note = "Use `CylindricalMediumProjection::min_sound_speed` instead"
    )]
    pub fn min_sound_speed(&self) -> f64 {
        self.sound_speed.iter().cloned().fold(f64::MAX, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_config_default() {
        let config = AxisymmetricConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    #[allow(deprecated)]
    fn test_config_hifu() {
        let config = AxisymmetricConfig::hifu_default();
        assert!(config.validate().is_ok());
        assert!(config.nonlinear);
    }

    #[test]
    #[allow(deprecated)]
    fn test_cfl_stability() {
        let config = AxisymmetricConfig::default();
        // With default settings and water sound speed
        assert!(config.is_stable(1500.0));
    }

    #[test]
    #[allow(deprecated)]
    fn test_homogeneous_medium() {
        let medium = AxisymmetricMedium::homogeneous(64, 32, 1500.0, 1000.0);
        assert_eq!(medium.sound_speed.dim(), (64, 32));
        assert_eq!(medium.max_sound_speed(), 1500.0);
    }
}
