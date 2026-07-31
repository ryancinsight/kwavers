use aequitas::systems::si::quantities::{Length, MolarConcentration, ReciprocalLength};
use aequitas::systems::si::units::{MicromolePerLiter, MolePerLiter, Nanometer};
use anyhow::{Context, Result};
use kwavers_analysis::signal_processing::spectroscopy::{SpectralUnmixer, SpectralUnmixingConfig};
use kwavers_optics::chromophores::HemoglobinDatabase;
use leto::{Array2, Array3};

/// Blood oxygenation map result
#[derive(Debug, Clone)]
pub struct OxygenationMap {
    /// Oxygen saturation map (0-1 range, fraction not percentage).
    pub so2_map: Array3<f64>,
    /// Oxyhemoglobin concentration map (mol/L at the spectral solver boundary).
    pub hbo2_concentration: Array3<f64>,
    /// Deoxyhemoglobin concentration map (mol/L at the spectral solver boundary).
    pub hb_concentration: Array3<f64>,
    /// Total hemoglobin concentration map (mol/L at the spectral solver boundary).
    pub total_hb_concentration: Array3<f64>,
    /// Residual error map (relative).
    pub residual_map: Array3<f64>,
    /// Wavelengths used.
    pub wavelengths: Vec<Length>,
}

/// Blood oxygenation estimation configuration
#[derive(Debug, Clone)]
pub struct OxygenationConfig {
    /// Wavelengths for spectral imaging.
    pub wavelengths: Vec<Length>,
    /// Spectral unmixing configuration
    pub unmixing_config: SpectralUnmixingConfig,
    /// Minimum total hemoglobin for valid sO₂.
    pub min_total_hb: MolarConcentration,
}

impl Default for OxygenationConfig {
    fn default() -> Self {
        Self {
            // Optimal wavelength selection for hemoglobin spectroscopy
            wavelengths: vec![
                Length::from_unit::<Nanometer>(532.0), // Green (strong Hb absorption)
                Length::from_unit::<Nanometer>(700.0), // Red (near isosbestic)
                Length::from_unit::<Nanometer>(800.0), // NIR window (HbO₂ peak)
                Length::from_unit::<Nanometer>(850.0), // NIR window (balanced)
            ],
            unmixing_config: SpectralUnmixingConfig::default(),
            min_total_hb: MolarConcentration::from_unit::<MicromolePerLiter>(10.0),
        }
    }
}

/// Estimate blood oxygenation from multi-wavelength absorption maps
///
/// # Arguments
///
/// - `absorption_maps`: Absorption coefficient maps at each wavelength (m⁻¹)
/// - `config`: Oxygenation estimation configuration
///
/// # Returns
///
/// Spatial maps of oxygen saturation and hemoglobin concentrations
/// # Errors
/// - Propagates any `KwaversError` returned by called functions.
///
pub fn estimate_oxygenation(
    absorption_maps: &[Array3<f64>],
    config: &OxygenationConfig,
) -> Result<OxygenationMap> {
    // Validate inputs
    if absorption_maps.len() != config.wavelengths.len() {
        anyhow::bail!(
            "Number of absorption maps ({}) does not match wavelength count ({})",
            absorption_maps.len(),
            config.wavelengths.len()
        );
    }

    if config.wavelengths.len() < 2 {
        anyhow::bail!("At least 2 wavelengths required for oxygenation estimation");
    }

    let wavelengths_nm: Vec<f64> = config
        .wavelengths
        .iter()
        .map(|wavelength| wavelength.in_unit::<Nanometer>())
        .collect();
    if wavelengths_nm
        .iter()
        .any(|&wavelength| !wavelength.is_finite() || wavelength <= 0.0)
    {
        anyhow::bail!("Optical wavelengths must be finite and positive");
    }

    let minimum_total_hb_molar = config.min_total_hb.in_unit::<MolePerLiter>();
    if !minimum_total_hb_molar.is_finite() || minimum_total_hb_molar <= 0.0 {
        anyhow::bail!("Minimum total hemoglobin must be finite and positive");
    }

    // Get spatial dimensions
    let [nx, ny, nz] = absorption_maps[0].shape();

    // Create hemoglobin database
    let hb_db = HemoglobinDatabase::standard();

    // Build extinction matrix for these wavelengths
    let n_wavelengths = config.wavelengths.len();
    let mut extinction_matrix = Array2::zeros((n_wavelengths, 2)); // 2 chromophores: HbO₂, Hb

    for (i, &wavelength_nm) in wavelengths_nm.iter().enumerate() {
        let (eps_hbo2, eps_hb) = hb_db
            .extinction_pair(wavelength_nm)
            .context(format!("Failed to get extinction at {wavelength_nm} nm"))?;

        // Convert from M⁻¹·cm⁻¹ to m⁻¹ per M concentration
        // μₐ = ln(10) · ε · C · 100, so ε_effective = ln(10) · ε · 100
        let factor = 2.303 * 100.0; // ln(10) * 100
        extinction_matrix[[i, 0]] = eps_hbo2 * factor;
        extinction_matrix[[i, 1]] = eps_hb * factor;
    }

    // Create spectral unmixer
    let chromophore_names = vec!["HbO₂".to_owned(), "Hb".to_owned()];
    let unmixer = SpectralUnmixer::new(
        extinction_matrix,
        wavelengths_nm,
        chromophore_names,
        config.unmixing_config.clone(),
    )?;

    // Perform volumetric unmixing
    let unmixing_result = unmixer.unmix_volumetric(absorption_maps)?;

    // Extract concentration maps
    let hbo2_concentration = unmixing_result.concentration_maps[0].clone();
    let hb_concentration = unmixing_result.concentration_maps[1].clone();

    // Compute total hemoglobin and oxygen saturation
    let mut so2_map = Array3::zeros((nx, ny, nz));
    let mut total_hb_concentration = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let hbo2 = hbo2_concentration[[i, j, k]];
                let hb = hb_concentration[[i, j, k]];
                let total = hbo2 + hb;

                total_hb_concentration[[i, j, k]] = total;

                // Compute sO₂ only where total Hb is above threshold
                if total >= minimum_total_hb_molar {
                    so2_map[[i, j, k]] = hbo2 / total;
                } else {
                    // Mark as invalid (NaN or 0)
                    so2_map[[i, j, k]] = 0.0;
                }
            }
        }
    }

    Ok(OxygenationMap {
        so2_map,
        hbo2_concentration,
        hb_concentration,
        total_hb_concentration,
        residual_map: unmixing_result.residual_map,
        wavelengths: config.wavelengths.clone(),
    })
}

/// Create arterial blood reference oxygenation (for validation)
///
/// Returns typical arterial blood properties at specified wavelengths
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn arterial_blood_reference(wavelengths: &[Length]) -> Result<Vec<ReciprocalLength>> {
    let hb_db = HemoglobinDatabase::standard();
    wavelengths
        .iter()
        .map(|wavelength| {
            hb_db
                .arterial_blood_absorption(wavelength.in_unit::<Nanometer>())
                .map(ReciprocalLength::from_base)
        })
        .collect()
}

/// Create venous blood reference oxygenation (for validation)
///
/// Returns typical venous blood properties at specified wavelengths
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn venous_blood_reference(wavelengths: &[Length]) -> Result<Vec<ReciprocalLength>> {
    let hb_db = HemoglobinDatabase::standard();
    wavelengths
        .iter()
        .map(|wavelength| {
            hb_db
                .venous_blood_absorption(wavelength.in_unit::<Nanometer>())
                .map(ReciprocalLength::from_base)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_contract_uses_typed_optical_units() {
        let config = OxygenationConfig::default();

        assert_eq!(config.wavelengths[0], Length::from_unit::<Nanometer>(532.0));
        assert_eq!(
            config.min_total_hb,
            MolarConcentration::from_unit::<MicromolePerLiter>(10.0)
        );
        assert_eq!(config.min_total_hb.in_unit::<MolePerLiter>(), 1.0e-5);
    }

    #[test]
    fn reference_absorption_returns_inverse_lengths() {
        let wavelengths = [
            Length::from_unit::<Nanometer>(532.0),
            Length::from_unit::<Nanometer>(800.0),
        ];
        let arterial = arterial_blood_reference(&wavelengths).expect("reference wavelengths");
        let venous = venous_blood_reference(&wavelengths).expect("reference wavelengths");

        assert_eq!(arterial.len(), wavelengths.len());
        assert_eq!(venous.len(), wavelengths.len());
        assert!(arterial.iter().all(|value| value.into_base().is_finite()));
        assert!(venous.iter().all(|value| value.into_base().is_finite()));
        assert!(arterial[0].into_base() > arterial[1].into_base());
        assert_ne!(arterial[1], venous[1]);
    }

    #[test]
    fn estimate_rejects_nonphysical_wavelengths() {
        let mut config = OxygenationConfig::default();
        config.wavelengths[0] = Length::from_unit::<Nanometer>(0.0);
        let maps = vec![Array3::zeros((1, 1, 1)); config.wavelengths.len()];

        let error = estimate_oxygenation(&maps, &config).expect_err("zero wavelength rejected");
        assert!(error.to_string().contains("finite and positive"));
    }
}
