//! Hemoglobin Chromophore Database

use anyhow::{Context, Result};
use super::spectrum::ExtinctionSpectrum;

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
    pub fn standard() -> Self {
        let hbo2_data = vec![
            (450, 106_112.0), (475, 88_872.0), (500, 64_568.0), (525, 40_944.0),
            (532, 35_464.0), (550, 24_424.0), (575, 17_304.0), (600, 11_800.0),
            (625, 8_168.0), (650, 5_824.0), (675, 4_784.0), (700, 4_448.0),
            (725, 4_656.0), (750, 5_144.0), (775, 5_904.0), (800, 6_896.0),
            (825, 8_136.0), (850, 9_632.0), (875, 11_360.0), (900, 13_328.0),
            (925, 15_536.0), (950, 17_984.0), (975, 20_672.0), (1000, 23_600.0),
        ];

        let hb_data = vec![
            (450, 112_736.0), (475, 97_088.0), (500, 78_272.0), (525, 60_096.0),
            (532, 54_664.0), (550, 43_760.0), (575, 31_352.0), (600, 20_944.0),
            (625, 13_600.0), (650, 9_440.0), (675, 7_632.0), (700, 7_072.0),
            (725, 7_376.0), (750, 8_176.0), (775, 9_472.0), (800, 11_264.0),
            (825, 13_552.0), (850, 16_336.0), (875, 19_616.0), (900, 23_392.0),
            (925, 27_664.0), (950, 32_432.0), (975, 37_696.0), (1000, 43_456.0),
        ];

        Self {
            hbo2: ExtinctionSpectrum::new("HbO₂ (Oxyhemoglobin)", hbo2_data),
            hb: ExtinctionSpectrum::new("Hb (Deoxyhemoglobin)", hb_data),
        }
    }

    pub fn hbo2_extinction(&self, wavelength_nm: f64) -> Result<f64> {
        self.hbo2.at_wavelength(wavelength_nm)
            .context("Failed to get HbO₂ extinction coefficient")
    }

    pub fn hb_extinction(&self, wavelength_nm: f64) -> Result<f64> {
        self.hb.at_wavelength(wavelength_nm)
            .context("Failed to get Hb extinction coefficient")
    }

    pub fn extinction_pair(&self, wavelength_nm: f64) -> Result<(f64, f64)> {
        let hbo2 = self.hbo2_extinction(wavelength_nm)?;
        let hb = self.hb_extinction(wavelength_nm)?;
        Ok((hbo2, hb))
    }

    pub fn absorption_coefficient(
        &self,
        wavelength_nm: f64,
        hbo2_concentration_molar: f64,
        hb_concentration_molar: f64,
    ) -> Result<f64> {
        let (eps_hbo2, eps_hb) = self.extinction_pair(wavelength_nm)?;
        let mu_a = 2.303 * (eps_hbo2 * hbo2_concentration_molar + eps_hb * hb_concentration_molar) * 100.0;
        Ok(mu_a)
    }

    pub fn oxygen_saturation(&self, hbo2_concentration: f64, hb_concentration: f64) -> Result<f64> {
        let total = hbo2_concentration + hb_concentration;
        if total <= 0.0 { anyhow::bail!("Total hemoglobin concentration must be positive"); }
        Ok(hbo2_concentration / total)
    }

    pub fn typical_blood_parameters() -> (f64, f64, f64) { (2.3e-3, 0.98, 0.75) }

    pub fn arterial_blood_absorption(&self, wavelength_nm: f64) -> Result<f64> {
        let (total, so2, _) = Self::typical_blood_parameters();
        self.absorption_coefficient(wavelength_nm, total * so2, total * (1.0 - so2))
    }

    pub fn venous_blood_absorption(&self, wavelength_nm: f64) -> Result<f64> {
        let (total, _, so2) = Self::typical_blood_parameters();
        self.absorption_coefficient(wavelength_nm, total * so2, total * (1.0 - so2))
    }

    pub fn isosbestic_points() -> Vec<u32> { vec![500, 545, 570, 584, 797] }
    pub fn hbo2_spectrum(&self) -> &ExtinctionSpectrum { &self.hbo2 }
    pub fn hb_spectrum(&self) -> &ExtinctionSpectrum { &self.hb }
}

impl Default for HemoglobinDatabase {
    fn default() -> Self { Self::standard() }
}
