//! Thermal module for bioheat transfer
//!
//! Implements the Pennes bioheat equation for modeling temperature rise
//! in tissue during ultrasound exposure.
//!
//! References:
//! - Pennes (1948) "Analysis of tissue and arterial blood temperatures"
//! - Nyborg (1988) "Solutions of the bio-heat transfer equation"
//! - ter Haar & Coussios (2007) "High intensity focused ultrasound"

// Fully documented module — enforce complete public-item docs (incremental
// crate-wide policy; see lib.rs).
#![deny(missing_docs)]

pub mod ablation;
pub mod coupling;
pub mod diffusion;
mod equivalent_minutes;
pub mod perfusion;
pub mod properties;
pub(crate) mod response;
pub mod thermal_dose;

pub use ablation::{AblationField, AblationKinetics, AblationState};
pub use coupling::{AcousticHeatingSource, TemperatureCoefficients, ThermalAcousticCoupling};
pub use diffusion::ThermalDiffusionConfig;
pub use equivalent_minutes::CumulativeEquivalentMinutes;
pub use thermal_dose::ThermalCEM43Grid;

// Re-export canonical domain type
pub use kwavers_medium::properties::ThermalPropertyData;

/// Common tissue types with thermal properties
///
/// These constructors return canonical `ThermalPropertyData` from the domain layer.
/// For Pennes solver simulations, also specify arterial temperature and metabolic heat
/// as separate simulation parameters.
pub mod tissues {
    use aequitas::systems::si::quantities::{
        MassDensity, MassDensityRate, SpecificHeatCapacity, ThermalConductivity,
    };
    use kwavers_core::constants::fundamental::DENSITY_TISSUE;
    use kwavers_core::constants::medical::BLOOD_SPECIFIC_HEAT;
    use kwavers_core::constants::tissue_acoustics::{
        DENSITY_BREAST_FAT, DENSITY_LIVER, DENSITY_MUSCLE,
    };
    use kwavers_core::constants::tissue_thermal::{
        SPECIFIC_HEAT_FAT, SPECIFIC_HEAT_LIVER, SPECIFIC_HEAT_MUSCLE, SPECIFIC_HEAT_TISSUE,
    };
    use kwavers_medium::properties::ThermalPropertyData;

    /// Liver tissue properties
    ///
    /// Reference: Duck (1990) "Physical Properties of Tissue"
    ///
    /// # Typical Simulation Parameters
    ///
    /// - Arterial temperature: 37.0°C
    /// - Metabolic heat: 33,800 W/m³ (high metabolic activity)
    /// # Panics
    /// - Panics if `Liver tissue properties are valid`.
    ///
    #[must_use]
    pub fn liver() -> ThermalPropertyData {
        ThermalPropertyData::new(
            ThermalConductivity::from_base(0.52),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_LIVER),
            MassDensity::from_base(DENSITY_LIVER),
            Some(MassDensityRate::from_base(16.7)),
            Some(SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT)),
        )
        .expect("Liver tissue properties are valid")
    }

    /// Muscle tissue properties
    ///
    /// Reference: Duck (1990) "Physical Properties of Tissue"
    ///
    /// # Typical Simulation Parameters
    ///
    /// - Arterial temperature: 37.0°C
    /// - Metabolic heat: 684 W/m³
    /// # Panics
    /// - Panics if `Muscle tissue properties are valid`.
    ///
    #[must_use]
    pub fn muscle() -> ThermalPropertyData {
        ThermalPropertyData::new(
            ThermalConductivity::from_base(0.49),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_MUSCLE),
            MassDensity::from_base(DENSITY_MUSCLE),
            Some(MassDensityRate::from_base(0.54)),
            Some(SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT)),
        )
        .expect("Muscle tissue properties are valid")
    }

    /// Fat tissue properties
    ///
    /// Reference: Duck (1990) "Physical Properties of Tissue"
    ///
    /// # Typical Simulation Parameters
    ///
    /// - Arterial temperature: 37.0°C
    /// - Metabolic heat: 400 W/m³
    /// # Panics
    /// - Panics if `Fat tissue properties are valid`.
    ///
    #[must_use]
    pub fn fat() -> ThermalPropertyData {
        ThermalPropertyData::new(
            ThermalConductivity::from_base(0.21),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_FAT),
            MassDensity::from_base(DENSITY_BREAST_FAT),
            Some(MassDensityRate::from_base(0.3)),
            Some(SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT)),
        )
        .expect("Fat tissue properties are valid")
    }

    /// Tumor tissue properties (hypoxic)
    ///
    /// Reference: Clinical hyperthermia literature
    ///
    /// # Typical Simulation Parameters
    ///
    /// - Arterial temperature: 37.0°C
    /// - Metabolic heat: 5,000 W/m³ (higher metabolism, poor perfusion)
    /// # Panics
    /// - Panics if `Tumor tissue properties are valid`.
    ///
    #[must_use]
    pub fn tumor() -> ThermalPropertyData {
        ThermalPropertyData::new(
            ThermalConductivity::from_base(0.55),
            SpecificHeatCapacity::from_base(SPECIFIC_HEAT_TISSUE),
            MassDensity::from_base(DENSITY_TISSUE),
            Some(MassDensityRate::from_base(0.2)),
            Some(SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT)),
        )
        .expect("Tumor tissue properties are valid")
    }

    /// Soft tissue properties (generic)
    ///
    /// This is an alias for the canonical domain constructor.
    ///
    /// # Typical Simulation Parameters
    ///
    /// - Arterial temperature: 37.0°C
    /// - Metabolic heat: 400 W/m³ (basal metabolism)
    #[must_use]
    pub fn soft_tissue() -> ThermalPropertyData {
        ThermalPropertyData::soft_tissue()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::DENSITY_TISSUE;
    use kwavers_core::constants::medical::BLOOD_SPECIFIC_HEAT;
    use kwavers_core::constants::tissue_acoustics::{
        DENSITY_BREAST_FAT, DENSITY_LIVER, DENSITY_MUSCLE,
    };

    #[test]
    fn test_tissue_constructors() {
        let liver = tissues::liver();
        assert_eq!(liver.conductivity().into_base(), 0.52);
        assert_eq!(liver.density().into_base(), DENSITY_LIVER);
        assert!(liver.has_bioheat_parameters());

        let muscle = tissues::muscle();
        assert_eq!(muscle.conductivity().into_base(), 0.49);
        assert_eq!(muscle.density().into_base(), DENSITY_MUSCLE);
        assert!(muscle.has_bioheat_parameters());

        let fat = tissues::fat();
        assert_eq!(fat.conductivity().into_base(), 0.21);
        assert_eq!(fat.density().into_base(), DENSITY_BREAST_FAT);
        assert!(fat.has_bioheat_parameters());

        let tumor = tissues::tumor();
        assert_eq!(tumor.conductivity().into_base(), 0.55);
        assert_eq!(tumor.density().into_base(), DENSITY_TISSUE);
        assert!(tumor.has_bioheat_parameters());

        let soft = tissues::soft_tissue();
        assert_eq!(soft.conductivity().into_base(), 0.5);
        assert_eq!(soft.density().into_base(), DENSITY_TISSUE);
        assert!(soft.has_bioheat_parameters());
    }

    #[test]
    fn test_thermal_diffusivity() {
        let liver = tissues::liver();
        let alpha = liver.thermal_diffusivity();

        // α = k / (ρc) = 0.52 / (1060 * 3540) ≈ 1.39e-7 m²/s
        let expected = liver.conductivity().into_base()
            / (liver.density().into_base() * liver.specific_heat().into_base());
        assert!((alpha.into_base() - expected).abs() < 1e-12);

        // Should be in reasonable range for tissue (10^-8 to 10^-6 m²/s)
        assert!(alpha.into_base() > 1e-8 && alpha.into_base() < 1e-6);
    }

    #[test]
    fn test_bioheat_parameters() {
        let tissue = tissues::liver();

        assert!(tissue.has_bioheat_parameters());
        let w_b = tissue.blood_perfusion.unwrap();
        let c_b = tissue.blood_specific_heat.unwrap();

        assert!(w_b.into_base() > 0.0);
        assert!(c_b.into_base() > 0.0);
        assert_eq!(c_b.into_base(), BLOOD_SPECIFIC_HEAT);
    }
}
