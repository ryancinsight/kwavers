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

use aequitas::systems::si::quantities::{
    MassDensity, MassDensityRate, SpecificHeatCapacity, ThermalConductivity, ThermalDiffusivity,
};
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
    pub blood_perfusion: Option<MassDensityRate<f64>>,

    /// Blood specific heat c_b (J/kg/K)
    ///
    /// Optional: Only for biological tissue in bio-heat equation.
    /// Typical value: ~3617 J/kg/K
    pub blood_specific_heat: Option<SpecificHeatCapacity<f64>>,
}

impl ThermalPropertyData {
    /// Construct with validation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        conductivity: ThermalConductivity<f64>,
        specific_heat: SpecificHeatCapacity<f64>,
        density: MassDensity<f64>,
        blood_perfusion: Option<MassDensityRate<f64>>,
        blood_specific_heat: Option<SpecificHeatCapacity<f64>>,
    ) -> Result<Self, String> {
        let thermophysical =
            ThermophysicalProperties::try_from_quantities(density, specific_heat, conductivity)
                .map_err(|error| error.to_string())?;

        Self::from_thermophysical(thermophysical, blood_perfusion, blood_specific_heat)
    }

    /// Construct from a validated Proteus thermophysical bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when conductivity is zero or a supplied bioheat
    /// parameter violates its physical domain.
    pub fn from_thermophysical(
        thermophysical: ThermophysicalProperties<f64>,
        blood_perfusion: Option<MassDensityRate<f64>>,
        blood_specific_heat: Option<SpecificHeatCapacity<f64>>,
    ) -> Result<Self, String> {
        if thermophysical.thermal_conductivity().quantity().as_base() == &0.0 {
            return Err("Thermal conductivity must be positive, got 0".to_owned());
        }
        if let Some(w_b) = blood_perfusion {
            if w_b.into_base() < 0.0 {
                return Err(format!(
                    "Blood perfusion must be non-negative, got {}",
                    w_b.into_base()
                ));
            }
        }
        if let Some(c_b) = blood_specific_heat {
            if c_b.into_base() <= 0.0 {
                return Err(format!(
                    "Blood specific heat must be positive, got {}",
                    c_b.into_base()
                ));
            }
        }

        Ok(Self {
            thermophysical,
            blood_perfusion,
            blood_specific_heat,
        })
    }

    /// Borrow the canonical Proteus thermophysical bundle.
    #[must_use]
    pub const fn thermophysical(&self) -> &ThermophysicalProperties<f64> {
        &self.thermophysical
    }

    /// Thermal conductivity k (W/m/K).
    #[must_use]
    pub fn conductivity(&self) -> ThermalConductivity<f64> {
        *self.thermophysical.thermal_conductivity().quantity()
    }

    /// Specific heat capacity c (J/kg/K).
    #[must_use]
    pub fn specific_heat(&self) -> SpecificHeatCapacity<f64> {
        *self.thermophysical.specific_heat_capacity().quantity()
    }

    /// Density rho (kg/m3).
    #[must_use]
    pub fn density(&self) -> MassDensity<f64> {
        *self.thermophysical.density().quantity()
    }

    /// Thermal diffusivity α = k/(ρc) (m²/s)
    #[inline]
    #[must_use]
    pub fn thermal_diffusivity(&self) -> ThermalDiffusivity<f64> {
        self.thermophysical.thermal_diffusivity()
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
            ThermalConductivity::from_base(THERMAL_CONDUCTIVITY_WATER),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_WATER),
            MassDensity::from_base(DENSITY_WATER),
            None,
            None,
        )
        .expect("water catalog properties satisfy the Proteus contract")
    }

    /// Soft tissue properties (generic)
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self::new(
            ThermalConductivity::from_base(0.5),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_TISSUE),
            MassDensity::from_base(DENSITY_TISSUE),
            Some(MassDensityRate::from_base(0.5)),
            Some(SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT)),
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
            ThermalConductivity::from_base(0.32),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_BONE), // 1313 J/(kg·K) (Duck 1990 Table 9.1)
            MassDensity::from_base(BONE_DENSITY),                // 1900 kg/m³ (Duck 1990)
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
            self.conductivity().into_base(),
            self.specific_heat().into_base(),
            self.density().into_base(),
            self.thermal_diffusivity().into_base()
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
        assert!((alpha.into_base() - expected).abs() < 1e-10);
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
        assert!(ThermalPropertyData::new(
            ThermalConductivity::from_base(-0.5),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_TISSUE),
            MassDensity::from_base(DENSITY_TISSUE),
            None,
            None,
        )
        .is_err());

        // Valid parameters should succeed
        let tp = ThermalPropertyData::new(
            ThermalConductivity::from_base(0.5),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_TISSUE),
            MassDensity::from_base(DENSITY_TISSUE),
            Some(MassDensityRate::from_base(0.5)),
            Some(SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT)),
        )
        .unwrap();
        assert_eq!(tp.conductivity().into_base(), 0.5);
    }
}
