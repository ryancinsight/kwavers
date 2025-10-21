//! Acoustic Radiation Force for Shear Wave Generation
//!
//! Implements acoustic radiation force impulse (ARFI) for generating shear waves
//! in soft tissue.
//!
//! ## Physics
//!
//! Acoustic radiation force arises from momentum transfer when ultrasound waves
//! are absorbed or reflected. For a focused ultrasound beam:
//!
//! F = (2αI)/c
//!
//! where:
//! - F is radiation force density (N/m³)
//! - α is absorption coefficient (Np/m)
//! - I is acoustic intensity (W/m²)
//! - c is sound speed (m/s)
//!
//! ## References
//!
//! - Nightingale, K., et al. (2002). "Acoustic radiation force impulse imaging."
//!   *Ultrasound in Medicine & Biology*, 28(2), 227-235.
//! - Palmeri, M. L., et al. (2005). "Ultrasonic tracking of acoustic radiation
//!   force-induced displacements." *IEEE TUFFC*, 52(8), 1300-1313.

use crate::error::{KwaversError, KwaversResult};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;
use std::f64::consts::PI;

/// Acoustic radiation force push pulse parameters
///
/// # Clinical Values
///
/// - Push duration: 50-400 μs (typ. 100-200 μs)
/// - Push frequency: 3-8 MHz (typ. 5 MHz)
/// - Focus depth: 20-80 mm
/// - F-number: 1.5-3.0 (typ. 2.0)
#[derive(Debug, Clone)]
pub struct PushPulseParameters {
    /// Push pulse frequency (Hz)
    pub frequency: f64,
    /// Push pulse duration (s)
    pub duration: f64,
    /// Peak acoustic intensity (W/m²)
    pub intensity: f64,
    /// Focal depth (m)
    pub focal_depth: f64,
    /// F-number (focal_depth / aperture_width)
    pub f_number: f64,
}

impl Default for PushPulseParameters {
    fn default() -> Self {
        Self {
            frequency: 5.0e6,  // 5 MHz
            duration: 150e-6,  // 150 μs
            intensity: 1000.0, // 1 kW/m²
            focal_depth: 0.04, // 40 mm
            f_number: 2.0,
        }
    }
}

impl PushPulseParameters {
    /// Create custom push pulse parameters
    ///
    /// # Arguments
    ///
    /// * `frequency` - Push frequency in Hz
    /// * `duration` - Push duration in seconds
    /// * `intensity` - Peak intensity in W/m²
    /// * `focal_depth` - Focal depth in meters
    /// * `f_number` - F-number (dimensionless)
    pub fn new(
        frequency: f64,
        duration: f64,
        intensity: f64,
        focal_depth: f64,
        f_number: f64,
    ) -> KwaversResult<Self> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "frequency".to_string(),
                    value: frequency,
                    reason: "must be positive".to_string(),
                },
            ));
        }
        if duration <= 0.0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "duration".to_string(),
                    value: duration,
                    reason: "must be positive".to_string(),
                },
            ));
        }
        if intensity <= 0.0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "intensity".to_string(),
                    value: intensity,
                    reason: "must be positive".to_string(),
                },
            ));
        }

        Ok(Self {
            frequency,
            duration,
            intensity,
            focal_depth,
            f_number,
        })
    }
}

/// Acoustic radiation force generator
#[derive(Debug)]
pub struct AcousticRadiationForce {
    /// Push pulse configuration
    parameters: PushPulseParameters,
    /// Medium sound speed (m/s)
    sound_speed: f64,
    /// Medium absorption coefficient (Np/m)
    absorption: f64,
    /// Medium density (kg/m³)
    density: f64,
    /// Computational grid
    grid: Grid,
}

impl AcousticRadiationForce {
    /// Create new acoustic radiation force generator
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `medium` - Tissue medium properties
    pub fn new(grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        // Get medium properties at center
        let (nx, ny, nz) = grid.dimensions();
        let ci = nx / 2;
        let cj = ny / 2;
        let ck = nz / 2;

        let sound_speed = medium.sound_speed(ci, cj, ck);
        let density = medium.density(ci, cj, ck);

        // Estimate absorption coefficient
        // For soft tissue at 5 MHz: α ≈ 0.5 dB/cm/MHz ≈ 5.8 Np/m
        let absorption = 5.8; // Np/m (simplified for now)

        Ok(Self {
            parameters: PushPulseParameters::default(),
            sound_speed,
            absorption,
            density,
            grid: grid.clone(),
        })
    }

    /// Set custom push pulse parameters
    pub fn set_parameters(&mut self, parameters: PushPulseParameters) {
        self.parameters = parameters;
    }

    /// Get current push pulse parameters
    #[must_use]
    pub fn parameters(&self) -> &PushPulseParameters {
        &self.parameters
    }

    /// Apply push pulse to generate shear wave
    ///
    /// # Arguments
    ///
    /// * `push_location` - Focal point [x, y, z] in meters
    ///
    /// # Returns
    ///
    /// Initial displacement field (3D array)
    ///
    /// # References
    ///
    /// Nightingale et al. (2002): Radiation force F = (2αI)/c generates
    /// initial displacement proportional to pulse duration and intensity.
    pub fn apply_push_pulse(&self, push_location: [f64; 3]) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut displacement = Array3::zeros((nx, ny, nz));

        // Calculate radiation force density
        // F = (2αI)/c (N/m³)
        let force_density = (2.0 * self.absorption * self.parameters.intensity) / self.sound_speed;

        // Calculate initial displacement from force impulse
        // u₀ = (F × Δt) / (ρ × ω₀)
        // where ω₀ is characteristic frequency
        let omega = 2.0 * PI * self.parameters.frequency;
        let displacement_scale =
            (force_density * self.parameters.duration) / (self.density * omega);

        // Calculate focal region dimensions
        // Lateral: FWHM ≈ 1.2 × λ × F-number
        // Axial: FWHM ≈ 6 × λ × F-number²
        let wavelength = self.sound_speed / self.parameters.frequency;
        let lateral_width = 1.2 * wavelength * self.parameters.f_number;
        let axial_length = 6.0 * wavelength * self.parameters.f_number * self.parameters.f_number;

        // Apply Gaussian-shaped displacement around focal point
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;

                    // Distance from focal point
                    let dx = x - push_location[0];
                    let dy = y - push_location[1];
                    let dz = z - push_location[2];

                    // Lateral and axial distances
                    let r_lateral = (dx * dx + dy * dy).sqrt();
                    let r_axial = dz.abs();

                    // Gaussian focal profile
                    let lateral_profile = (-4.0 * (r_lateral / lateral_width).powi(2)).exp();
                    let axial_profile = (-4.0 * (r_axial / axial_length).powi(2)).exp();

                    displacement[[i, j, k]] = displacement_scale * lateral_profile * axial_profile;
                }
            }
        }

        Ok(displacement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn test_push_parameters_default() {
        let params = PushPulseParameters::default();
        assert_eq!(params.frequency, 5.0e6);
        assert_eq!(params.duration, 150e-6);
        assert!((params.f_number - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_push_parameters_validation() {
        let result = PushPulseParameters::new(-1.0, 100e-6, 1000.0, 0.04, 2.0);
        assert!(result.is_err());

        let result = PushPulseParameters::new(5e6, -100e-6, 1000.0, 0.04, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_radiation_force_creation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let result = AcousticRadiationForce::new(&grid, &medium);
        assert!(result.is_ok());
    }

    #[test]
    fn test_push_pulse_generation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let arf = AcousticRadiationForce::new(&grid, &medium).unwrap();

        let push_location = [0.025, 0.025, 0.025];
        let displacement = arf.apply_push_pulse(push_location).unwrap();

        // Check displacement field properties
        assert_eq!(displacement.dim(), (50, 50, 50));

        // Maximum displacement should be at or near focal point
        let max_disp = displacement
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_disp > 0.0, "Maximum displacement should be positive");

        // Displacement should decay away from focal point
        let corner_disp = displacement[[0, 0, 0]];
        assert!(
            corner_disp < max_disp * 0.1,
            "Displacement should be localized"
        );
    }
}
