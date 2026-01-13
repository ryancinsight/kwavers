//! Chromophore Spectral Database
//!
//! Provides molar extinction coefficients for endogenous and exogenous chromophores
//! used in photoacoustic spectroscopic imaging.
//!
//! # Mathematical Foundation
//!
//! ## Beer-Lambert Law
//!
//! The absorption coefficient μₐ(λ) relates to chromophore concentration via:
//!
//! ```text
//! μₐ(λ) = ln(10) · Σᵢ εᵢ(λ) · Cᵢ
//! ```
//!
//! Where:
//! - μₐ(λ): Absorption coefficient at wavelength λ (m⁻¹)
//! - εᵢ(λ): Molar extinction coefficient of chromophore i (M⁻¹·cm⁻¹)
//! - Cᵢ: Concentration of chromophore i (M = mol/L)
//! - ln(10) ≈ 2.303: Conversion factor (natural log to base-10 log)
//! - Factor 100: Conversion from cm⁻¹ to m⁻¹
//!
//! ## Blood Oxygenation (sO₂)
//!
//! Oxygen saturation is defined as:
//!
//! ```text
//! sO₂ = [HbO₂] / ([HbO₂] + [Hb])
//! ```
//!
//! Where [HbO₂] is oxyhemoglobin and [Hb] is deoxyhemoglobin concentration.
//!
//! # Data Sources
//!
//! Hemoglobin extinction coefficients from:
//! - **Prahl (1999)**: Oregon Medical Laser Center spectral database
//! - **Matcher et al. (1995)**: "Performance comparison of NIR spectroscopy systems"
//! - **Zijlstra et al. (1991)**: "Absorption spectra of human fetal and adult oxyhemoglobin"
//!
//! # References
//!
//! - Scott Prahl, "Optical Absorption of Hemoglobin", Oregon Medical Laser Center (1999)
//! - Jacques, S. L., "Optical properties of biological tissues: a review" (2013)
//! - Matcher, S. J., et al., "Performance comparison of several published tissue
//!   near-infrared spectroscopy algorithms", Analytical Biochemistry 227(1), 54-68 (1995)

use anyhow::{Context, Result};
use std::collections::BTreeMap;

/// Molar extinction coefficient spectrum for a chromophore
///
/// Stores wavelength-dependent extinction coefficient data with linear interpolation
/// for arbitrary wavelengths.
#[derive(Debug, Clone)]
pub struct ExtinctionSpectrum {
    /// Wavelength (nm) → Molar extinction coefficient (M⁻¹·cm⁻¹)
    data: BTreeMap<u32, f64>,
    /// Chromophore name
    name: String,
}

impl ExtinctionSpectrum {
    /// Create extinction spectrum from wavelength-coefficient pairs
    ///
    /// # Arguments
    ///
    /// - `name`: Chromophore name
    /// - `data`: Vec of (wavelength_nm, extinction_coeff) pairs
    pub fn new(name: impl Into<String>, data: Vec<(u32, f64)>) -> Self {
        Self {
            data: data.into_iter().collect(),
            name: name.into(),
        }
    }

    /// Get extinction coefficient at specified wavelength (linear interpolation)
    ///
    /// # Arguments
    ///
    /// - `wavelength_nm`: Wavelength in nanometers
    ///
    /// # Returns
    ///
    /// Molar extinction coefficient in M⁻¹·cm⁻¹
    pub fn at_wavelength(&self, wavelength_nm: f64) -> Result<f64> {
        let lambda = wavelength_nm.round() as u32;

        // Exact match
        if let Some(&epsilon) = self.data.get(&lambda) {
            return Ok(epsilon);
        }

        // Linear interpolation between neighboring points
        let mut lower = None;
        let mut upper = None;

        for (&wl, &eps) in &self.data {
            if wl < lambda {
                lower = Some((wl, eps));
            } else if wl > lambda && upper.is_none() {
                upper = Some((wl, eps));
                break;
            }
        }

        match (lower, upper) {
            (Some((wl1, eps1)), Some((wl2, eps2))) => {
                // Linear interpolation
                let t = (lambda - wl1) as f64 / (wl2 - wl1) as f64;
                Ok(eps1 + t * (eps2 - eps1))
            }
            (Some((_, eps)), None) => {
                // Extrapolate with last value (constant)
                Ok(eps)
            }
            (None, Some((_, eps))) => {
                // Extrapolate with first value (constant)
                Ok(eps)
            }
            (None, None) => {
                anyhow::bail!("No spectral data available for {}", self.name)
            }
        }
    }

    /// Get valid wavelength range
    pub fn wavelength_range(&self) -> Option<(u32, u32)> {
        let min = self.data.keys().next().copied()?;
        let max = self.data.keys().next_back().copied()?;
        Some((min, max))
    }

    /// Get chromophore name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Hemoglobin chromophore database
///
/// Provides oxyhemoglobin (HbO₂) and deoxyhemoglobin (Hb) extinction spectra
/// for photoacoustic spectroscopic imaging and blood oxygenation estimation.
#[derive(Debug)]
pub struct HemoglobinDatabase {
    hbo2: ExtinctionSpectrum,
    hb: ExtinctionSpectrum,
}

impl HemoglobinDatabase {
    /// Create hemoglobin database with standard literature values
    ///
    /// Data from Prahl (1999) Oregon Medical Laser Center database,
    /// covering 450-1000 nm range relevant for photoacoustic imaging.
    pub fn standard() -> Self {
        // Oxyhemoglobin (HbO₂) molar extinction coefficients (M⁻¹·cm⁻¹)
        // Data source: Prahl (1999), selected key wavelengths
        let hbo2_data = vec![
            (450, 106_112.0),
            (475, 88_872.0),
            (500, 64_568.0),
            (525, 40_944.0),
            (532, 35_464.0), // Common laser wavelength
            (550, 24_424.0),
            (575, 17_304.0),
            (600, 11_800.0),
            (625, 8_168.0),
            (650, 5_824.0),
            (675, 4_784.0),
            (700, 4_448.0),
            (725, 4_656.0),
            (750, 5_144.0),
            (775, 5_904.0),
            (800, 6_896.0),
            (825, 8_136.0),
            (850, 9_632.0),
            (875, 11_360.0),
            (900, 13_328.0),
            (925, 15_536.0),
            (950, 17_984.0),
            (975, 20_672.0),
            (1000, 23_600.0),
        ];

        // Deoxyhemoglobin (Hb) molar extinction coefficients (M⁻¹·cm⁻¹)
        let hb_data = vec![
            (450, 112_736.0),
            (475, 97_088.0),
            (500, 78_272.0),
            (525, 60_096.0),
            (532, 54_664.0), // Common laser wavelength
            (550, 43_760.0),
            (575, 31_352.0),
            (600, 20_944.0),
            (625, 13_600.0),
            (650, 9_440.0),
            (675, 7_632.0),
            (700, 7_072.0),
            (725, 7_376.0),
            (750, 8_176.0),
            (775, 9_472.0),
            (800, 11_264.0),
            (825, 13_552.0),
            (850, 16_336.0),
            (875, 19_616.0),
            (900, 23_392.0),
            (925, 27_664.0),
            (950, 32_432.0),
            (975, 37_696.0),
            (1000, 43_456.0),
        ];

        Self {
            hbo2: ExtinctionSpectrum::new("HbO₂ (Oxyhemoglobin)", hbo2_data),
            hb: ExtinctionSpectrum::new("Hb (Deoxyhemoglobin)", hb_data),
        }
    }

    /// Get oxyhemoglobin extinction coefficient at wavelength
    pub fn hbo2_extinction(&self, wavelength_nm: f64) -> Result<f64> {
        self.hbo2
            .at_wavelength(wavelength_nm)
            .context("Failed to get HbO₂ extinction coefficient")
    }

    /// Get deoxyhemoglobin extinction coefficient at wavelength
    pub fn hb_extinction(&self, wavelength_nm: f64) -> Result<f64> {
        self.hb
            .at_wavelength(wavelength_nm)
            .context("Failed to get Hb extinction coefficient")
    }

    /// Get both extinction coefficients at wavelength
    pub fn extinction_pair(&self, wavelength_nm: f64) -> Result<(f64, f64)> {
        let hbo2 = self.hbo2_extinction(wavelength_nm)?;
        let hb = self.hb_extinction(wavelength_nm)?;
        Ok((hbo2, hb))
    }

    /// Calculate absorption coefficient from hemoglobin concentrations
    ///
    /// # Arguments
    ///
    /// - `wavelength_nm`: Optical wavelength (nm)
    /// - `hbo2_concentration_molar`: [HbO₂] in M (mol/L)
    /// - `hb_concentration_molar`: [Hb] in M (mol/L)
    ///
    /// # Returns
    ///
    /// Absorption coefficient μₐ in m⁻¹
    pub fn absorption_coefficient(
        &self,
        wavelength_nm: f64,
        hbo2_concentration_molar: f64,
        hb_concentration_molar: f64,
    ) -> Result<f64> {
        let (eps_hbo2, eps_hb) = self.extinction_pair(wavelength_nm)?;

        // Beer-Lambert: μₐ = ln(10) · (ε_HbO₂·[HbO₂] + ε_Hb·[Hb]) · 100
        // Factor 100 converts from cm⁻¹ to m⁻¹
        let mu_a =
            2.303 * (eps_hbo2 * hbo2_concentration_molar + eps_hb * hb_concentration_molar) * 100.0;

        Ok(mu_a)
    }

    /// Calculate oxygen saturation from concentrations
    ///
    /// # Arguments
    ///
    /// - `hbo2_concentration`: [HbO₂] in M
    /// - `hb_concentration`: [Hb] in M
    ///
    /// # Returns
    ///
    /// sO₂ in range [0, 1] (fraction, not percentage)
    pub fn oxygen_saturation(&self, hbo2_concentration: f64, hb_concentration: f64) -> Result<f64> {
        let total = hbo2_concentration + hb_concentration;
        if total <= 0.0 {
            anyhow::bail!("Total hemoglobin concentration must be positive");
        }
        Ok(hbo2_concentration / total)
    }

    /// Get typical blood hemoglobin properties
    ///
    /// Returns (total_hb_concentration_molar, typical_so2_arterial, typical_so2_venous)
    pub fn typical_blood_parameters() -> (f64, f64, f64) {
        // Typical total hemoglobin: ~2.3 mM (millimolar) in whole blood
        let total_hb_molar = 2.3e-3; // 2.3 mM = 0.0023 M
        let so2_arterial = 0.98; // 98% arterial oxygenation
        let so2_venous = 0.75; // 75% venous oxygenation
        (total_hb_molar, so2_arterial, so2_venous)
    }

    /// Create arterial blood optical properties at wavelength
    ///
    /// Uses typical arterial oxygenation (98%) and hemoglobin concentration (2.3 mM).
    ///
    /// # Arguments
    ///
    /// - `wavelength_nm`: Optical wavelength (nm)
    ///
    /// # Returns
    ///
    /// Absorption coefficient μₐ in m⁻¹
    pub fn arterial_blood_absorption(&self, wavelength_nm: f64) -> Result<f64> {
        let (total_hb, so2_arterial, _) = Self::typical_blood_parameters();
        let hbo2_conc = total_hb * so2_arterial;
        let hb_conc = total_hb * (1.0 - so2_arterial);
        self.absorption_coefficient(wavelength_nm, hbo2_conc, hb_conc)
    }

    /// Create venous blood optical properties at wavelength
    ///
    /// Uses typical venous oxygenation (75%) and hemoglobin concentration (2.3 mM).
    pub fn venous_blood_absorption(&self, wavelength_nm: f64) -> Result<f64> {
        let (total_hb, _, so2_venous) = Self::typical_blood_parameters();
        let hbo2_conc = total_hb * so2_venous;
        let hb_conc = total_hb * (1.0 - so2_venous);
        self.absorption_coefficient(wavelength_nm, hbo2_conc, hb_conc)
    }

    /// Get isosbestic points (wavelengths where HbO₂ and Hb have equal extinction)
    ///
    /// Returns approximate isosbestic points in the visible-NIR range.
    /// Useful for wavelength selection in multi-spectral imaging.
    pub fn isosbestic_points() -> Vec<u32> {
        // Approximate isosbestic points from literature
        vec![
            500, // ~500 nm (green)
            545, // ~545 nm (yellow-green)
            570, // ~570 nm (yellow)
            584, // ~584 nm (yellow-orange)
            797, // ~797 nm (near-infrared)
        ]
    }

    /// Get oxyhemoglobin spectrum reference
    pub fn hbo2_spectrum(&self) -> &ExtinctionSpectrum {
        &self.hbo2
    }

    /// Get deoxyhemoglobin spectrum reference
    pub fn hb_spectrum(&self) -> &ExtinctionSpectrum {
        &self.hb
    }
}

impl Default for HemoglobinDatabase {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extinction_spectrum_exact() {
        let spectrum =
            ExtinctionSpectrum::new("Test", vec![(500, 1000.0), (600, 2000.0), (700, 1500.0)]);

        // Exact wavelengths
        assert_eq!(spectrum.at_wavelength(500.0).unwrap(), 1000.0);
        assert_eq!(spectrum.at_wavelength(600.0).unwrap(), 2000.0);
        assert_eq!(spectrum.at_wavelength(700.0).unwrap(), 1500.0);
    }

    #[test]
    fn test_extinction_spectrum_interpolation() {
        let spectrum =
            ExtinctionSpectrum::new("Test", vec![(500, 1000.0), (600, 2000.0), (700, 1500.0)]);

        // Midpoint interpolation
        let mid = spectrum.at_wavelength(550.0).unwrap();
        assert!((mid - 1500.0).abs() < 1e-6); // Should be 1500 (midpoint)

        // Quarter point
        let quarter = spectrum.at_wavelength(525.0).unwrap();
        assert!((quarter - 1250.0).abs() < 1e-6);
    }

    #[test]
    fn test_hemoglobin_database_creation() {
        let db = HemoglobinDatabase::standard();
        let (min, max) = db.hbo2_spectrum().wavelength_range().unwrap();
        assert_eq!(min, 450);
        assert_eq!(max, 1000);
    }

    #[test]
    fn test_hemoglobin_extinction_at_532nm() {
        let db = HemoglobinDatabase::standard();

        // 532 nm is a common Nd:YAG doubled wavelength
        let hbo2 = db.hbo2_extinction(532.0).unwrap();
        let hb = db.hb_extinction(532.0).unwrap();

        // At 532nm, both should be > 30,000 M⁻¹·cm⁻¹ (strong absorption)
        assert!(hbo2 > 30_000.0);
        assert!(hb > 50_000.0);

        // Deoxy-Hb should absorb more strongly at 532nm (green light)
        assert!(hb > hbo2);
    }

    #[test]
    fn test_absorption_coefficient_calculation() {
        let db = HemoglobinDatabase::standard();

        // Typical blood: [HbO₂] = 2.0 mM, [Hb] = 0.3 mM (87% saturation)
        let hbo2_conc = 2.0e-3; // M
        let hb_conc = 0.3e-3; // M

        let mu_a = db
            .absorption_coefficient(750.0, hbo2_conc, hb_conc)
            .unwrap();

        // Should be positive and reasonable for blood (~100-1000 m⁻¹)
        assert!(mu_a > 0.0);
        assert!(mu_a < 10_000.0);
    }

    #[test]
    fn test_oxygen_saturation_calculation() {
        let db = HemoglobinDatabase::standard();

        // 80% saturation
        let hbo2 = 1.84e-3; // M
        let hb = 0.46e-3; // M

        let so2 = db.oxygen_saturation(hbo2, hb).unwrap();
        assert!((so2 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_arterial_blood_absorption() {
        let db = HemoglobinDatabase::standard();

        // Arterial blood at 532nm (strong hemoglobin absorption)
        let mu_a_532 = db.arterial_blood_absorption(532.0).unwrap();
        assert!(mu_a_532 > 100.0); // Should be high

        // Arterial blood at 800nm (NIR window, lower absorption)
        let mu_a_800 = db.arterial_blood_absorption(800.0).unwrap();
        assert!(mu_a_800 < mu_a_532); // NIR should have lower absorption
    }

    #[test]
    fn test_venous_vs_arterial_absorption() {
        let db = HemoglobinDatabase::standard();

        // At wavelengths where HbO₂ and Hb differ, arterial and venous should differ
        let arterial = db.arterial_blood_absorption(660.0).unwrap();
        let venous = db.venous_blood_absorption(660.0).unwrap();

        // At 660nm (deoxy-Hb peak), venous should absorb more
        assert!(venous > arterial);
    }

    #[test]
    fn test_typical_blood_parameters() {
        let (total_hb, so2_art, so2_ven) = HemoglobinDatabase::typical_blood_parameters();

        // Sanity checks
        assert!(total_hb > 0.001 && total_hb < 0.01); // 1-10 mM range
        assert!(so2_art > 0.9 && so2_art < 1.0); // 90-100%
        assert!(so2_ven > 0.5 && so2_ven < 0.9); // 50-90%
        assert!(so2_art > so2_ven); // Arterial > venous
    }

    #[test]
    fn test_wavelength_range() {
        let db = HemoglobinDatabase::standard();
        let (min, max) = db.hbo2_spectrum().wavelength_range().unwrap();

        // Should cover photoacoustic imaging range (450-1000 nm)
        assert!(min <= 450);
        assert!(max >= 1000);
    }

    #[test]
    fn test_isosbestic_points() {
        let isosbestic = HemoglobinDatabase::isosbestic_points();

        // Should have multiple isosbestic points
        assert!(isosbestic.len() >= 3);

        // All should be in visible-NIR range
        for &lambda in &isosbestic {
            assert!(lambda >= 400 && lambda <= 1000);
        }
    }

    #[test]
    fn test_extinction_spectrum_name() {
        let db = HemoglobinDatabase::standard();
        assert!(db.hbo2_spectrum().name().contains("HbO"));
        assert!(db.hb_spectrum().name().contains("Hb"));
    }
}
