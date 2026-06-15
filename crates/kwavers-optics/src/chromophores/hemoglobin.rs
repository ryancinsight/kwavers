//! Hemoglobin Chromophore Database

use super::spectrum::ExtinctionSpectrum;
use anyhow::{Context, Result};

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
    /// Create hemoglobin database with standard literature values.
    ///
    /// ## Units and Normalisation
    ///
    /// Values are molar extinction coefficients in M⁻¹·cm⁻¹ for the hemoglobin
    /// **tetramer** — the per-heme OMLC/Prahl values × 4, since a tetramer
    /// carries four heme groups. Concentrations are therefore tetramer-molar, to
    /// match `typical_blood_parameters()` (whole-blood [Hb] ≈ 2.3 mmol/L
    /// tetramer; tetramer molar mass ≈ 64 500 g/mol). Beer-Lambert:
    ///
    /// ```text
    /// μ_a = ln(10) × Σ_i ε_i [M⁻¹·cm⁻¹] × c_i [mol/L] × 100 [cm/m]   →  [m⁻¹]
    /// ```
    ///
    /// Spectral check: the HbO₂ and Hb curves cross at the isosbestic points
    /// (`isosbestic_points()`). In the red (e.g. 650 nm) deoxy-Hb absorbs far
    /// more than HbO₂ (ε_Hb ≈ 10× ε_HbO₂); in the near-infrared (> 800 nm) HbO₂
    /// exceeds Hb — the basis of pulse oximetry. Near 800 nm the two are
    /// approximately equal (the isosbestic point at ≈ 797 nm).
    ///
    /// ## Source
    ///
    /// Prahl SA (1999), *Optical Absorption of Hemoglobin*, OMLC compiled
    /// tabulation (Gratzer / Kollias), per-heme molar extinction scaled ×4 to
    /// the tetramer. <https://omlc.org/spectra/hemoglobin/summary.html>
    #[must_use]
    pub fn standard() -> Self {
        // Per-tetramer = per-heme Prahl/OMLC molar extinction × 4.
        let hbo2_data = vec![
            (450, 251_264.0),
            (475, 120_454.4),
            (500, 83_731.2),
            (525, 117_076.8),
            (532, 175_504.0),
            (550, 172_064.0),
            (575, 196_688.0),
            (600, 12_800.0),
            (625, 3_096.0),
            (650, 1_472.0),
            (675, 1_142.4),
            (700, 1_160.0),
            (725, 1_560.0),
            (750, 2_072.0),
            (775, 2_708.8),
            (800, 3_264.0),
            (825, 3_825.6),
            (850, 4_232.0),
            (875, 4_550.4),
            (900, 4_792.0),
            (925, 4_907.2),
            (950, 4_816.0),
            (975, 4_576.0),
            (1000, 4_096.0),
        ];

        let hb_data = vec![
            (450, 413_168.0),
            (475, 60_193.6),
            (500, 83_448.0),
            (525, 137_590.4),
            (532, 162_336.0),
            (550, 213_648.0),
            (575, 173_360.0),
            (600, 58_708.8),
            (625, 23_627.2),
            (650, 15_000.48),
            (675, 10_510.56),
            (700, 7_177.12),
            (725, 4_408.8),
            (750, 5_620.96),
            (775, 4_852.0),
            (800, 3_046.88),
            (825, 2_773.28),
            (850, 2_765.28),
            (875, 2_856.32),
            (900, 3_047.36),
            (925, 3_089.44),
            (950, 2_408.96),
            (975, 1_557.152),
            (1000, 827.136),
        ];

        Self {
            hbo2: ExtinctionSpectrum::new("HbO₂ (Oxyhemoglobin)", hbo2_data),
            hb: ExtinctionSpectrum::new("Hb (Deoxyhemoglobin)", hb_data),
        }
    }
    /// Hbo2 extinction.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn hbo2_extinction(&self, wavelength_nm: f64) -> Result<f64> {
        self.hbo2
            .at_wavelength(wavelength_nm)
            .context("Failed to get HbO₂ extinction coefficient")
    }
    /// Hb extinction.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn hb_extinction(&self, wavelength_nm: f64) -> Result<f64> {
        self.hb
            .at_wavelength(wavelength_nm)
            .context("Failed to get Hb extinction coefficient")
    }
    /// Extinction pair.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn extinction_pair(&self, wavelength_nm: f64) -> Result<(f64, f64)> {
        let hbo2 = self.hbo2_extinction(wavelength_nm)?;
        let hb = self.hb_extinction(wavelength_nm)?;
        Ok((hbo2, hb))
    }
    /// Absorption coefficient.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn absorption_coefficient(
        &self,
        wavelength_nm: f64,
        hbo2_concentration_molar: f64,
        hb_concentration_molar: f64,
    ) -> Result<f64> {
        let (eps_hbo2, eps_hb) = self.extinction_pair(wavelength_nm)?;
        let mu_a = 2.303
            * eps_hbo2.mul_add(hbo2_concentration_molar, eps_hb * hb_concentration_molar)
            * 100.0;
        Ok(mu_a)
    }
    /// Oxygen saturation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn oxygen_saturation(&self, hbo2_concentration: f64, hb_concentration: f64) -> Result<f64> {
        let total = hbo2_concentration + hb_concentration;
        if total <= 0.0 {
            anyhow::bail!("Total hemoglobin concentration must be positive");
        }
        Ok(hbo2_concentration / total)
    }
    /// Typical blood parameters.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn typical_blood_parameters() -> (f64, f64, f64) {
        (2.3e-3, 0.98, 0.75)
    }
    /// Arterial blood absorption.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn arterial_blood_absorption(&self, wavelength_nm: f64) -> Result<f64> {
        let (total, so2, _) = Self::typical_blood_parameters();
        self.absorption_coefficient(wavelength_nm, total * so2, total * (1.0 - so2))
    }
    /// Venous blood absorption.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn venous_blood_absorption(&self, wavelength_nm: f64) -> Result<f64> {
        let (total, _, so2) = Self::typical_blood_parameters();
        self.absorption_coefficient(wavelength_nm, total * so2, total * (1.0 - so2))
    }

    #[must_use]
    pub fn isosbestic_points() -> Vec<u32> {
        vec![500, 545, 570, 584, 797]
    }
    #[must_use]
    pub fn hbo2_spectrum(&self) -> &ExtinctionSpectrum {
        &self.hbo2
    }
    #[must_use]
    pub fn hb_spectrum(&self) -> &ExtinctionSpectrum {
        &self.hb
    }
}

impl Default for HemoglobinDatabase {
    fn default() -> Self {
        Self::standard()
    }
}
