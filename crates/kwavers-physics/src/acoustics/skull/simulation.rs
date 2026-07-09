use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::Array3;

use super::aberration::AberrationCorrection;
use super::analytical::{generate_ellipsoidal_skull, generate_spherical_skull};
use super::heterogeneous::HeterogeneousSkull;
use super::properties::AcousticSkullProperties;

/// Transcranial focused ultrasound simulation workflow
///
/// # Example
///
/// ```no_run
/// use kwavers_physics::acoustics::skull::{TranscranialSimulation, AcousticSkullProperties};
/// use kwavers_grid::Grid;
///
/// # fn example() -> kwavers_core::error::KwaversResult<()> {
/// let grid = Grid::new(200, 200, 200, 0.5e-3, 0.5e-3, 0.5e-3)?;
/// let skull_props = AcousticSkullProperties::default();
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
    pub skull_props: AcousticSkullProperties,
    /// Skull geometry mask (1 = skull, 0 = soft tissue)
    pub skull_mask: Option<Array3<f64>>,
    /// Heterogeneous skull model
    pub heterogeneous: Option<HeterogeneousSkull>,
}

impl TranscranialSimulation {
    /// Create new transcranial simulation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: &Grid, skull_props: AcousticSkullProperties) -> KwaversResult<Self> {
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
        use kwavers_imaging::medical::{CTImageLoader, MedicalImageLoader};
        let mut loader = CTImageLoader::new();
        let ct_data = leto::Array3::try_from(loader.load(ct_path)?)
            .map_err(|err| KwaversError::Shape(err.to_string()))?;

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
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn set_analytical_geometry(
        &mut self,
        model_type: &str,
        parameters: &[f64],
    ) -> KwaversResult<()> {
        let mask = match model_type {
            "sphere" => {
                generate_spherical_skull(&self.grid, self.skull_props.thickness, parameters[0])?
            }
            "ellipsoid" => {
                generate_ellipsoidal_skull(&self.grid, self.skull_props.thickness, parameters)?
            }
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compute_aberration_correction(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        let heterogeneous = self
            .heterogeneous
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Skull geometry not loaded".to_owned()))?;

        let correction = AberrationCorrection::new(&self.grid, heterogeneous);
        correction.compute_time_reversal_phases(frequency)
    }

    /// Estimate insertion loss through skull
    ///
    /// Returns expected pressure reduction factor
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// Estimate one-way pressure insertion loss through the skull.
    ///
    /// Models a single acoustic path from water through the skull and back into
    /// water (water→skull→water interfaces).
    ///
    /// # Pressure attenuation
    /// - Bulk absorption: `exp(−α f_MHz × d)` where `d` = skull thickness.
    /// - Interface reflection losses: through two interfaces the pressure
    ///   transmission coefficient of a water–skull–water slab is
    ///   `T_p = T_p(w→s) × T_p(s→w) = 2Z_skull/(Z_w+Z_s) × 2Z_w/(Z_w+Z_s)
    ///         = 4 Z_w Z_s/(Z_w+Z_s)² = T_I`
    ///   i.e. exactly the intensity transmission coefficient (always ≤ 1).
    ///
    /// Previous code used `2 × thickness` (round-trip) for attenuation but
    /// only a single `√T_I` for interfaces; both were wrong for a one-way path.
    ///
    /// # Returns
    /// Pressure amplitude ratio in [0, 1].
    pub fn estimate_insertion_loss(&self, frequency: f64) -> KwaversResult<f64> {
        let attenuation_np_per_m = self.skull_props.attenuation_at_frequency(frequency);

        // One-way path through skull (not round-trip)
        let attenuation_np = attenuation_np_per_m * self.skull_props.thickness;

        // Pressure amplitude after bulk absorption
        let amplitude_ratio = (-attenuation_np).exp();

        // Pressure TC through water→skull→water slab equals T_I (see docstring).
        // Z_water = ρ_water · c_water (sourced from SSOT constants)
        let water_z = DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM;
        let t_i = self.skull_props.transmission_coefficient(water_z);

        Ok(amplitude_ratio * t_i)
    }
}
