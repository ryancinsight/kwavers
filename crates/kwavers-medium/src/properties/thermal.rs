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

use aequitas::systems::si::quantities::{MassDensity, SpecificHeatCapacity, ThermalConductivity};
use kwavers_core::constants::acoustic_parameters::BONE_DENSITY;
use kwavers_core::constants::fundamental::{DENSITY_TISSUE, DENSITY_WATER};
use kwavers_core::constants::medical::BLOOD_SPECIFIC_HEAT;
use kwavers_core::constants::thermodynamic::{SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER};
use kwavers_core::constants::tissue_thermal::{SPECIFIC_HEAT_BONE, SPECIFIC_HEAT_TISSUE};
use proteus::ThermophysicalProperties;
use std::fmt;

/// Canonical thermal material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalPropertyData {
    thermophysical: ThermophysicalProperties<f64>,

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        conductivity: f64,
        specific_heat: f64,
        density: f64,
        blood_perfusion: Option<f64>,
        blood_specific_heat: Option<f64>,
    ) -> Result<Self, String> {
        let thermophysical = ThermophysicalProperties::try_from_quantities(
            MassDensity::from_base(density),
            SpecificHeatCapacity::from_base(specific_heat),
            ThermalConductivity::from_base(conductivity),
        )
        .map_err(|error| error.to_string())?;
        if conductivity == 0.0 {
            return Err("Thermal conductivity must be positive, got 0".to_owned());
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
            thermophysical,
            blood_perfusion,
            blood_specific_heat,
        })
    }

    /// Thermal conductivity k (W/m/K).
    #[must_use]
    pub fn conductivity(&self) -> f64 {
        self.thermophysical
            .thermal_conductivity()
            .quantity()
            .into_base()
    }

    /// Specific heat capacity c (J/kg/K).
    #[must_use]
    pub fn specific_heat(&self) -> f64 {
        self.thermophysical
            .specific_heat_capacity()
            .quantity()
            .into_base()
    }

    /// Density rho (kg/m3).
    #[must_use]
    pub fn density(&self) -> f64 {
        self.thermophysical.density().quantity().into_base()
    }

    /// Thermal diffusivity α = k/(ρc) (m²/s)
    #[inline]
    #[must_use]
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermophysical.thermal_diffusivity().into_base()
    }

    /// Check if bio-heat parameters are present
    #[inline]
    #[must_use]
    pub fn has_bioheat_parameters(&self) -> bool {
        self.blood_perfusion.is_some() && self.blood_specific_heat.is_some()
    }

    /// Water properties (at 20°C)
    #[must_use]
    pub fn water() -> Self {
        Self::new(
            THERMAL_CONDUCTIVITY_WATER,
            SPECIFIC_HEAT_WATER,
            DENSITY_WATER,
            None,
            None,
        )
        .expect("water catalog properties satisfy the Proteus contract")
    }

    /// Soft tissue properties (generic)
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self::new(
            0.5,
            SPECIFIC_HEAT_TISSUE,
            DENSITY_TISSUE,
            Some(0.5),
            Some(BLOOD_SPECIFIC_HEAT),
        )
        .expect("soft-tissue catalog properties satisfy the Proteus contract")
    }

    /// Bone properties
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn bone() -> Self {
        Self::new(
            0.32,
            SPECIFIC_HEAT_BONE, // 1313 J/(kg·K) (Duck 1990 Table 9.1)
            BONE_DENSITY,       // 1900 kg/m³ (Duck 1990)
            None,
            None,
        )
        .expect("bone catalog properties satisfy the Proteus contract")
    }
}

impl fmt::Display for ThermalPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Thermal(k={:.2} W/m/K, c={:.0} J/kg/K, ρ={:.0} kg/m³, α={:.2e} m²/s",
            self.conductivity(),
            self.specific_heat(),
            self.density(),
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
    use kwavers_core::constants::fundamental::DENSITY_TISSUE;

    #[test]
    fn test_thermal_diffusivity() {
        let water = ThermalPropertyData::water();
        let alpha = water.thermal_diffusivity();
        let expected = THERMAL_CONDUCTIVITY_WATER / (DENSITY_WATER * SPECIFIC_HEAT_WATER);
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
        use kwavers_core::constants::medical::BLOOD_SPECIFIC_HEAT;
        use kwavers_core::constants::tissue_thermal::SPECIFIC_HEAT_TISSUE;
        // Negative conductivity should fail
        assert!(
            ThermalPropertyData::new(-0.5, SPECIFIC_HEAT_TISSUE, DENSITY_TISSUE, None, None)
                .is_err()
        );

        // Valid parameters should succeed
        let tp = ThermalPropertyData::new(
            0.5,
            SPECIFIC_HEAT_TISSUE,
            DENSITY_TISSUE,
            Some(0.5),
            Some(BLOOD_SPECIFIC_HEAT),
        )
        .unwrap();
        assert_eq!(tp.conductivity(), 0.5);
    }
}
