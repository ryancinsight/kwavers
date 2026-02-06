use crate::clinical::imaging::chromophores::HemoglobinDatabase;
use crate::clinical::imaging::spectroscopy::{SpectralUnmixer, SpectralUnmixingConfig};
use anyhow::{Context, Result};
use ndarray::{Array2, Array3};

/// Blood oxygenation map result
#[derive(Debug, Clone)]
pub struct OxygenationMap {
    /// Oxygen saturation map (0-1 range, fraction not percentage)
    pub so2_map: Array3<f64>,
    /// Oxyhemoglobin concentration map (M)
    pub hbo2_concentration: Array3<f64>,
    /// Deoxyhemoglobin concentration map (M)
    pub hb_concentration: Array3<f64>,
    /// Total hemoglobin concentration map (M)
    pub total_hb_concentration: Array3<f64>,
    /// Residual error map (relative)
    pub residual_map: Array3<f64>,
    /// Wavelengths used (nm)
    pub wavelengths: Vec<f64>,
}

/// Blood oxygenation estimation configuration
#[derive(Debug, Clone)]
pub struct OxygenationConfig {
    /// Wavelengths for spectral imaging (nm)
    pub wavelengths: Vec<f64>,
    /// Spectral unmixing configuration
    pub unmixing_config: SpectralUnmixingConfig,
    /// Minimum total hemoglobin for valid sO₂ (M)
    pub min_total_hb: f64,
}

impl Default for OxygenationConfig {
    fn default() -> Self {
        Self {
            // Optimal wavelength selection for hemoglobin spectroscopy
            wavelengths: vec![
                532.0, // Green (strong Hb absorption)
                700.0, // Red (near isosbestic)
                800.0, // NIR window (HbO₂ peak)
                850.0, // NIR window (balanced)
            ],
            unmixing_config: SpectralUnmixingConfig::default(),
            min_total_hb: 1e-5, // 10 μM minimum (background threshold)
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

    // Get spatial dimensions
    let (nx, ny, nz) = absorption_maps[0].dim();

    // Create hemoglobin database
    let hb_db = HemoglobinDatabase::standard();

    // Build extinction matrix for these wavelengths
    let n_wavelengths = config.wavelengths.len();
    let mut extinction_matrix = Array2::zeros((n_wavelengths, 2)); // 2 chromophores: HbO₂, Hb

    for (i, &wavelength) in config.wavelengths.iter().enumerate() {
        let (eps_hbo2, eps_hb) = hb_db
            .extinction_pair(wavelength)
            .context(format!("Failed to get extinction at {} nm", wavelength))?;

        // Convert from M⁻¹·cm⁻¹ to m⁻¹ per M concentration
        // μₐ = ln(10) · ε · C · 100, so ε_effective = ln(10) · ε · 100
        let factor = 2.303 * 100.0; // ln(10) * 100
        extinction_matrix[[i, 0]] = eps_hbo2 * factor;
        extinction_matrix[[i, 1]] = eps_hb * factor;
    }

    // Create spectral unmixer
    let chromophore_names = vec!["HbO₂".to_string(), "Hb".to_string()];
    let unmixer = SpectralUnmixer::new(
        extinction_matrix,
        config.wavelengths.clone(),
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
                if total >= config.min_total_hb {
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
pub fn arterial_blood_reference(wavelengths: &[f64]) -> Result<Vec<f64>> {
    let hb_db = HemoglobinDatabase::standard();
    wavelengths
        .iter()
        .map(|&wl| hb_db.arterial_blood_absorption(wl))
        .collect()
}

/// Create venous blood reference oxygenation (for validation)
///
/// Returns typical venous blood properties at specified wavelengths
pub fn venous_blood_reference(wavelengths: &[f64]) -> Result<Vec<f64>> {
    let hb_db = HemoglobinDatabase::standard();
    wavelengths
        .iter()
        .map(|&wl| hb_db.venous_blood_absorption(wl))
        .collect()
}
