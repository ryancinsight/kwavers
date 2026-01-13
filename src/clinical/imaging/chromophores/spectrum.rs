//! Chromophore Extinction Spectrum Implementation

use anyhow::Result;
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
    pub fn new(name: impl Into<String>, data: Vec<(u32, f64)>) -> Self {
        Self {
            data: data.into_iter().collect(),
            name: name.into(),
        }
    }

    /// Get extinction coefficient at specified wavelength (linear interpolation)
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
                let t = (lambda - wl1) as f64 / (wl2 - wl1) as f64;
                Ok(eps1 + t * (eps2 - eps1))
            }
            (Some((_, eps)), None) => Ok(eps),
            (None, Some((_, eps))) => Ok(eps),
            (None, None) => {
                anyhow::bail!("No spectral data available for {}", self.name)
            }
        }
    }

    pub fn wavelength_range(&self) -> Option<(u32, u32)> {
        let min = self.data.keys().next().copied()?;
        let max = self.data.keys().next_back().copied()?;
        Some((min, max))
    }

    pub fn name(&self) -> &str { &self.name }
}
