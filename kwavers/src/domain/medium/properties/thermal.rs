//! Thermal material property data structures
//!
//! # Mathematical Foundation
//!
//! ## Heat Equation (Fourier's Law)
//!
//! ```text
//! ρc ∂T/∂t = ∇·(k∇T) + Q
//! ```
//!
//! Where:
//! - `T`: Temperature (K)
//! - `k`: Thermal conductivity (W/m/K)
//! - `ρ`: Density (kg/m³)
//! - `c`: Specific heat capacity (J/kg/K)
//! - `Q`: Heat source (W/m³)
//!
//! ## Bio-Heat Equation (Pennes)
//!
//! For biological tissue:
//! ```text
//! ρc ∂T/∂t = ∇·(k∇T) + w_b c_b (T_b - T) + Q_met + Q_ext
//! ```
//!
//! Additional terms:
//! - `w_b`: Blood perfusion rate (kg/m³/s)
//! - `c_b`: Blood specific heat (J/kg/K)
//! - `T_b`: Arterial blood temperature (K)
//!
//! ## Thermal Diffusivity
//!
//! ```text
//! α = k/(ρc)  (m²/s)
//! ```
//!
//! ## Invariants
//!
//! - `conductivity > 0`
//! - `specific_heat > 0`
//! - `density > 0`
//! - `blood_perfusion ≥ 0` (if present)
//! - `blood_specific_heat > 0` (if present)

use std::fmt;

/// Canonical thermal material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalPropertyData {
    /// Thermal conductivity k (W/m/K)
    ///
    /// Physical range:
    /// - Insulators: 0.01-1 W/m/K
    /// - Water: ~0.6 W/m/K
    /// - Tissue: 0.4-0.6 W/m/K
    /// - Metals: 10-400 W/m/K
    pub conductivity: f64,

    /// Specific heat capacity c (J/kg/K)
    ///
    /// Physical range:
    /// - Metals: 100-1000 J/kg/K
    /// - Water: ~4180 J/kg/K
    /// - Tissue: 3000-4000 J/kg/K
    pub specific_heat: f64,

    /// Density ρ (kg/m³)
    pub density: f64,

    /// Blood perfusion rate w_b (kg/m³/s)
    ///
    /// Optional: Only for biological tissue in bio-heat equation.
    /// Typical range: 0.5-10 kg/m³/s
    pub blood_perfusion: Option<f64>,

    /// Blood specific heat c_b (J/kg/K)
    ///
    /// Optional: Only for biological tissue in bio-heat equation.
    /// Typical value: ~3617 J/kg/K
    pub blood_specific_heat: Option<f64>,
}

impl ThermalPropertyData {
    /// Construct with validation
    pub fn new(
        conductivity: f64,
        specific_heat: f64,
        density: f64,
        blood_perfusion: Option<f64>,
        blood_specific_heat: Option<f64>,
    ) -> Result<Self, String> {
        if conductivity <= 0.0 {
            return Err(format!(
                "Thermal conductivity must be positive, got {}",
                conductivity
            ));
        }
        if specific_heat <= 0.0 {
            return Err(format!(
                "Specific heat must be positive, got {}",
                specific_heat
            ));
        }
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if let Some(w_b) = blood_perfusion {
            if w_b < 0.0 {
                return Err(format!("Blood perfusion must be non-negative, got {}", w_b));
            }
        }
        if let Some(c_b) = blood_specific_heat {
            if c_b <= 0.0 {
                return Err(format!("Blood specific heat must be positive, got {}", c_b));
            }
        }

        Ok(Self {
            conductivity,
            specific_heat,
            density,
            blood_perfusion,
            blood_specific_heat,
        })
    }

    /// Thermal diffusivity α = k/(ρc) (m²/s)
    #[inline]
    pub fn thermal_diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }

    /// Check if bio-heat parameters are present
    #[inline]
    pub fn has_bioheat_parameters(&self) -> bool {
        self.blood_perfusion.is_some() && self.blood_specific_heat.is_some()
    }

    /// Water properties (at 20°C)
    pub fn water() -> Self {
        Self {
            conductivity: 0.598,
            specific_heat: 4182.0,
            density: 998.0,
            blood_perfusion: None,
            blood_specific_heat: None,
        }
    }

    /// Soft tissue properties (generic)
    pub fn soft_tissue() -> Self {
        Self {
            conductivity: 0.5,
            specific_heat: 3600.0,
            density: 1050.0,
            blood_perfusion: Some(0.5),
            blood_specific_heat: Some(3617.0),
        }
    }

    /// Bone properties
    pub fn bone() -> Self {
        Self {
            conductivity: 0.32,
            specific_heat: 1300.0,
            density: 1850.0,
            blood_perfusion: None,
            blood_specific_heat: None,
        }
    }
}

impl fmt::Display for ThermalPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Thermal(k={:.2} W/m/K, c={:.0} J/kg/K, ρ={:.0} kg/m³, α={:.2e} m²/s",
            self.conductivity,
            self.specific_heat,
            self.density,
            self.thermal_diffusivity()
        )?;
        if self.has_bioheat_parameters() {
            write!(f, ", bio-heat enabled")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_diffusivity() {
        let water = ThermalPropertyData::water();
        let alpha = water.thermal_diffusivity();
        let expected = 0.598 / (998.0 * 4182.0);
        assert!((alpha - expected).abs() < 1e-10);
    }

    #[test]
    fn test_thermal_bioheat_detection() {
        let water = ThermalPropertyData::water();
        assert!(!water.has_bioheat_parameters());

        let tissue = ThermalPropertyData::soft_tissue();
        assert!(tissue.has_bioheat_parameters());
    }

    #[test]
    fn test_thermal_validation() {
        // Negative conductivity should fail
        assert!(ThermalPropertyData::new(-0.5, 3600.0, 1050.0, None, None).is_err());

        // Valid parameters should succeed
        assert!(ThermalPropertyData::new(0.5, 3600.0, 1050.0, Some(0.5), Some(3617.0)).is_ok());
    }
}
