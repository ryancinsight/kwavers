use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::Array3;

use super::aberration::AberrationCorrection;
use super::heterogeneous::HeterogeneousSkull;
use super::properties::SkullProperties;
use super::analytical::{generate_spherical_skull, generate_ellipsoidal_skull};

/// Transcranial focused ultrasound simulation workflow
///
/// # Example
///
/// ```no_run
/// use kwavers::physics::acoustics::skull::{TranscranialSimulation, SkullProperties};
/// use kwavers::domain::grid::Grid;
///
/// # fn example() -> kwavers::core::error::KwaversResult<()> {
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
    pub skull_props: SkullProperties,
    /// Skull geometry mask (1 = skull, 0 = soft tissue)
    pub skull_mask: Option<Array3<f64>>,
    /// Heterogeneous skull model
    pub heterogeneous: Option<HeterogeneousSkull>,
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
        use crate::domain::imaging::medical::{CTImageLoader, MedicalImageLoader};
        let mut loader = CTImageLoader::new();
        let ct_data = loader.load(ct_path)?;
        
        self.skull_mask = Some(HeterogeneousSkull::generate_mask_from_ct(&ct_data));
        self.heterogeneous = Some(HeterogeneousSkull::from_ct(&ct_data, &self.skull_props)?);
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
            "sphere" => generate_spherical_skull(&self.grid, self.skull_props.thickness, parameters[0])?,
            "ellipsoid" => generate_ellipsoidal_skull(&self.grid, self.skull_props.thickness, parameters)?,
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
        let heterogeneous = self
            .heterogeneous
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Skull geometry not loaded".to_string()))?;

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
}
