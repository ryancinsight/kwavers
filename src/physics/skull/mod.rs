//! Transcranial Focused Ultrasound (tFUS) - Skull Heterogeneity Modeling
//!
//! This module implements skull propagation and aberration correction for
//! transcranial ultrasound applications following clinical standards.
//!
//! ## Overview
//!
//! Skull bone introduces significant challenges for transcranial ultrasound:
//! 1. **High Acoustic Impedance**: Causes strong reflections (~80%)
//! 2. **Phase Aberrations**: Spatially varying thickness and density
//! 3. **Attenuation**: Frequency-dependent energy loss
//! 4. **Shear Wave Conversion**: Mode conversion at interfaces
//!
//! ## Literature References
//!
//! - Clement, G. T., & Hynynen, K. (2002). "A non-invasive method for focusing
//!   ultrasound through the skull." *Physics in Medicine & Biology*, 47(8), 1219.
//! - Aubry, J. F., et al. (2003). "Experimental demonstration of noninvasive
//!   transskull adaptive focusing." *IEEE TUFFC*, 50(10), 1128-1138.
//! - Marquet, F., et al. (2009). "Non-invasive transcranial ultrasound therapy
//!   based on a 3D CT scan." *Physics in Medicine & Biology*, 54(9), 2597.
//! - Pinton, G., et al. (2012). "Attenuation, scattering, and absorption of
//!   ultrasound in the skull bone." *Medical Physics*, 39(1), 299-307.
//!
//! ## Clinical Applications
//!
//! - Essential tremor treatment
//! - Parkinson's disease therapy
//! - Brain tumor ablation
//! - Blood-brain barrier opening
//! - Neuromodulation

pub mod aberration;
pub mod attenuation;
pub mod ct_based;
pub mod heterogeneous;

use crate::error::{KwaversError, KwaversResult};
use crate::grid::Grid;
use ndarray::Array3;

pub use aberration::AberrationCorrection;
pub use attenuation::SkullAttenuation;
pub use ct_based::CTBasedSkullModel;
pub use heterogeneous::HeterogeneousSkull;

/// Skull material properties based on literature
///
/// Reference: Pinton et al. (2012) "Attenuation, scattering, and absorption
/// of ultrasound in the skull bone"
#[derive(Debug, Clone)]
pub struct SkullProperties {
    /// Sound speed in skull (m/s) - typically 2800-3500 m/s
    pub sound_speed: f64,
    /// Density (kg/m³) - typically 1850-2000 kg/m³
    pub density: f64,
    /// Attenuation coefficient (Np/m/MHz) - typically 40-100
    pub attenuation_coeff: f64,
    /// Skull thickness (m) - typically 3-10 mm
    pub thickness: f64,
    /// Shear wave speed (m/s) - typically 1400-1800 m/s
    pub shear_speed: Option<f64>,
}

impl Default for SkullProperties {
    fn default() -> Self {
        // Typical adult skull properties
        Self {
            sound_speed: 3100.0,      // m/s (cortical bone)
            density: 1900.0,          // kg/m³
            attenuation_coeff: 60.0,  // Np/m/MHz
            thickness: 0.007,         // 7 mm average
            shear_speed: Some(1600.0), // m/s
        }
    }
}

impl SkullProperties {
    /// Create skull properties for specific bone types
    ///
    /// # Arguments
    ///
    /// * `bone_type` - "cortical", "trabecular", or "suture"
    pub fn from_bone_type(bone_type: &str) -> KwaversResult<Self> {
        match bone_type {
            "cortical" => Ok(Self {
                sound_speed: 3100.0,
                density: 1900.0,
                attenuation_coeff: 60.0,
                thickness: 0.007,
                shear_speed: Some(1600.0),
            }),
            "trabecular" => Ok(Self {
                sound_speed: 2400.0,
                density: 1600.0,
                attenuation_coeff: 40.0,
                thickness: 0.005,
                shear_speed: Some(1200.0),
            }),
            "suture" => Ok(Self {
                sound_speed: 1800.0,
                density: 1200.0,
                attenuation_coeff: 20.0,
                thickness: 0.002,
                shear_speed: None, // Soft tissue-like
            }),
            _ => Err(KwaversError::InvalidInput(format!(
                "Unknown bone type: {}",
                bone_type
            ))),
        }
    }

    /// Calculate acoustic impedance (Z = ρc)
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.sound_speed
    }

    /// Calculate transmission coefficient for normal incidence
    ///
    /// Reference: Kinsler et al. (2000) "Fundamentals of Acoustics"
    pub fn transmission_coefficient(&self, water_impedance: f64) -> f64 {
        let z_skull = self.acoustic_impedance();
        let t = 2.0 * water_impedance / (water_impedance + z_skull);
        t.powi(2) // Intensity transmission coefficient
    }

    /// Calculate attenuation for given frequency (Hz)
    ///
    /// Attenuation = α(f) = α₀ × f^n where n ≈ 1 for bone
    pub fn attenuation_at_frequency(&self, frequency: f64) -> f64 {
        let freq_mhz = frequency / 1e6;
        self.attenuation_coeff * freq_mhz
    }
}

/// Transcranial focused ultrasound simulation workflow
///
/// # Example
///
/// ```no_run
/// use kwavers::physics::skull::{TranscranialSimulation, SkullProperties};
/// use kwavers::grid::Grid;
///
/// # fn example() -> kwavers::error::KwaversResult<()> {
/// let grid = Grid::new(200, 200, 200, 0.5e-3, 0.5e-3, 0.5e-3)?;
/// let skull_props = SkullProperties::default();
///
/// let mut tfus = TranscranialSimulation::new(&grid, skull_props)?;
///
/// // Set analytical sphere geometry
/// tfus.set_analytical_geometry("sphere", &[20.0])?;
///
/// // Compute aberration correction
/// let phase_corrections = tfus.compute_aberration_correction(650e3)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct TranscranialSimulation {
    /// Computational grid
    grid: Grid,
    /// Skull material properties
    skull_props: SkullProperties,
    /// Skull geometry mask (1 = skull, 0 = soft tissue)
    skull_mask: Option<Array3<f64>>,
    /// Heterogeneous skull model
    heterogeneous: Option<HeterogeneousSkull>,
}

impl TranscranialSimulation {
    /// Create new transcranial simulation
    pub fn new(grid: &Grid, skull_props: SkullProperties) -> KwaversResult<Self> {
        Ok(Self {
            grid: grid.clone(),
            skull_props,
            skull_mask: None,
            heterogeneous: None,
        })
    }

    /// Load skull geometry from CT scan
    ///
    /// # Arguments
    ///
    /// * `ct_path` - Path to CT NIFTI file
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be loaded or format is invalid
    pub fn load_ct_geometry(&mut self, ct_path: &str) -> KwaversResult<()> {
        let ct_model = CTBasedSkullModel::from_file(ct_path)?;
        self.skull_mask = Some(ct_model.generate_mask(&self.grid)?);
        self.heterogeneous = Some(ct_model.to_heterogeneous(&self.grid)?);
        Ok(())
    }

    /// Set skull geometry from analytical model (sphere, ellipsoid, etc.)
    ///
    /// # Arguments
    ///
    /// * `model_type` - "sphere", "ellipsoid", or "realistic"
    /// * `parameters` - Model-specific parameters
    pub fn set_analytical_geometry(
        &mut self,
        model_type: &str,
        parameters: &[f64],
    ) -> KwaversResult<()> {
        let mask = match model_type {
            "sphere" => self.generate_spherical_skull(parameters[0])?,
            "ellipsoid" => self.generate_ellipsoidal_skull(parameters)?,
            _ => {
                return Err(KwaversError::InvalidInput(format!(
                    "Unknown model type: {}",
                    model_type
                )))
            }
        };

        let het = HeterogeneousSkull::from_mask(&self.grid, &mask, &self.skull_props)?;
        self.skull_mask = Some(mask);
        self.heterogeneous = Some(het);
        Ok(())
    }

    /// Compute aberration correction phases
    ///
    /// Uses time-reversal or pseudo-inverse methods to compute phase
    /// corrections that compensate for skull-induced aberrations.
    ///
    /// Reference: Aubry et al. (2003) IEEE TUFFC
    pub fn compute_aberration_correction(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        let heterogeneous = self.heterogeneous.as_ref().ok_or_else(|| {
            KwaversError::InvalidInput("Skull geometry not loaded".to_string())
        })?;

        let correction = AberrationCorrection::new(&self.grid, heterogeneous);
        correction.compute_time_reversal_phases(frequency)
    }

    /// Estimate insertion loss through skull
    ///
    /// Returns expected pressure reduction factor
    pub fn estimate_insertion_loss(&self, frequency: f64) -> KwaversResult<f64> {
        let attenuation_np_per_m = self.skull_props.attenuation_at_frequency(frequency);

        // Two-way path through skull
        let total_path = 2.0 * self.skull_props.thickness;
        let attenuation_np = attenuation_np_per_m * total_path;

        // Convert to amplitude ratio
        let amplitude_ratio = (-attenuation_np).exp();

        // Include reflection losses
        let water_z = 1.5e6; // Water impedance (kg/m²/s)
        let transmission = self.skull_props.transmission_coefficient(water_z);

        Ok(amplitude_ratio * transmission.sqrt())
    }

    // Private helper methods

    fn generate_spherical_skull(&self, radius: f64) -> KwaversResult<Array3<f64>> {
        let mut mask = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        let cx = self.grid.nx as f64 / 2.0;
        let cy = self.grid.ny as f64 / 2.0;
        let cz = self.grid.nz as f64 / 2.0;

        let inner_radius = radius - self.skull_props.thickness / self.grid.dx;
        let outer_radius = radius;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let r = ((i as f64 - cx).powi(2)
                        + (j as f64 - cy).powi(2)
                        + (k as f64 - cz).powi(2))
                    .sqrt();

                    if r >= inner_radius && r <= outer_radius {
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        Ok(mask)
    }

    fn generate_ellipsoidal_skull(&self, params: &[f64]) -> KwaversResult<Array3<f64>> {
        if params.len() < 3 {
            return Err(KwaversError::InvalidInput(
                "Ellipsoid requires 3 radii".to_string(),
            ));
        }

        let mut mask = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        let cx = self.grid.nx as f64 / 2.0;
        let cy = self.grid.ny as f64 / 2.0;
        let cz = self.grid.nz as f64 / 2.0;

        let (rx, ry, rz) = (params[0], params[1], params[2]);
        let thickness_x = self.skull_props.thickness / self.grid.dx;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let dx = (i as f64 - cx) / rx;
                    let dy = (j as f64 - cy) / ry;
                    let dz = (k as f64 - cz) / rz;

                    let r = (dx * dx + dy * dy + dz * dz).sqrt();

                    let inner_r = 1.0 - thickness_x / rx;
                    if r >= inner_r && r <= 1.0 {
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skull_properties_default() {
        let props = SkullProperties::default();
        assert!(props.sound_speed > 2800.0 && props.sound_speed < 3500.0);
        assert!(props.density > 1800.0 && props.density < 2100.0);
    }

    #[test]
    fn test_bone_types() {
        let cortical = SkullProperties::from_bone_type("cortical").unwrap();
        let trabecular = SkullProperties::from_bone_type("trabecular").unwrap();

        assert!(cortical.sound_speed > trabecular.sound_speed);
        assert!(cortical.density > trabecular.density);
    }

    #[test]
    fn test_acoustic_impedance() {
        let props = SkullProperties::default();
        let z = props.acoustic_impedance();

        // Skull impedance should be much higher than water (1.5 MRayl)
        assert!(z > 5.0e6);
        assert!(z < 8.0e6);
    }

    #[test]
    fn test_transmission_coefficient() {
        let props = SkullProperties::default();
        let water_z = 1.5e6;
        let t = props.transmission_coefficient(water_z);

        // Should have significant loss (<50% transmission)
        assert!(t > 0.0 && t < 0.5);
    }

    #[test]
    fn test_frequency_dependent_attenuation() {
        let props = SkullProperties::default();

        let atten_500k = props.attenuation_at_frequency(500e3);
        let atten_1m = props.attenuation_at_frequency(1e6);

        // Attenuation should increase with frequency
        assert!(atten_1m > atten_500k);
    }

    #[test]
    fn test_transcranial_simulation_creation() {
        let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001).unwrap();
        let props = SkullProperties::default();

        let sim = TranscranialSimulation::new(&grid, props);
        assert!(sim.is_ok());
    }

    #[test]
    fn test_analytical_sphere_geometry() {
        let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
        let props = SkullProperties::default();

        let mut sim = TranscranialSimulation::new(&grid, props).unwrap();
        let result = sim.set_analytical_geometry("sphere", &[20.0]);

        assert!(result.is_ok());
        assert!(sim.skull_mask.is_some());
    }

    #[test]
    fn test_insertion_loss_estimation() {
        let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001).unwrap();
        let props = SkullProperties::default();

        let sim = TranscranialSimulation::new(&grid, props).unwrap();
        let loss = sim.estimate_insertion_loss(650e3).unwrap();

        // Should have significant insertion loss
        assert!(loss > 0.1 && loss < 0.5);
    }
}
