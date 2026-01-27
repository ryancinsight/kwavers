//! Thermal module for bioheat transfer
//!
//! Implements the Pennes bioheat equation for modeling temperature rise
//! in tissue during ultrasound exposure.
//!
//! References:
//! - Pennes (1948) "Analysis of tissue and arterial blood temperatures"
//! - Nyborg (1988) "Solutions of the bio-heat transfer equation"
//! - ter Haar & Coussios (2007) "High intensity focused ultrasound"

pub mod diffusion;
pub mod perfusion;
pub mod properties;
pub mod thermal_dose;

pub use diffusion::ThermalDiffusionConfig;
pub use thermal_dose::ThermalDose;

// Re-export canonical domain type
pub use crate::domain::medium::properties::ThermalPropertyData;

/// Common tissue types with thermal properties
///
/// These constructors return canonical `ThermalPropertyData` from the domain layer.
/// For Pennes solver simulations, also specify arterial temperature and metabolic heat
/// as separate simulation parameters.
pub mod tissues {
    use crate::domain::medium::properties::ThermalPropertyData;

    /// Liver tissue properties
    ///
    /// Reference: Duck (1990) "Physical Properties of Tissue"
    ///
    /// # Typical Simulation Parameters
    ///
    /// - Arterial temperature: 37.0°C
    /// - Metabolic heat: 33,800 W/m³ (high metabolic activity)
    #[must_use]
    pub fn liver() -> ThermalPropertyData {
        ThermalPropertyData::new(
            0.52,         // conductivity (W/m/K)
            3540.0,       // specific_heat (J/kg/K)
            1060.0,       // density (kg/m³)
            Some(16.7),   // blood_perfusion (kg/m³/s) - high perfusion
            Some(3617.0), // blood_specific_heat (J/kg/K)
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
    #[must_use]
    pub fn muscle() -> ThermalPropertyData {
        ThermalPropertyData::new(
            0.49,         // conductivity (W/m/K)
            3421.0,       // specific_heat (J/kg/K)
            1090.0,       // density (kg/m³)
            Some(0.54),   // blood_perfusion (kg/m³/s)
            Some(3617.0), // blood_specific_heat (J/kg/K)
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
    #[must_use]
    pub fn fat() -> ThermalPropertyData {
        ThermalPropertyData::new(
            0.21,         // conductivity (W/m/K)
            2348.0,       // specific_heat (J/kg/K)
            911.0,        // density (kg/m³)
            Some(0.3),    // blood_perfusion (kg/m³/s) - low perfusion
            Some(3617.0), // blood_specific_heat (J/kg/K)
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
    #[must_use]
    pub fn tumor() -> ThermalPropertyData {
        ThermalPropertyData::new(
            0.55,         // conductivity (W/m/K)
            3600.0,       // specific_heat (J/kg/K)
            1050.0,       // density (kg/m³)
            Some(0.2),    // blood_perfusion (kg/m³/s) - poor perfusion
            Some(3617.0), // blood_specific_heat (J/kg/K)
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

    #[test]
    fn test_tissue_constructors() {
        let liver = tissues::liver();
        assert_eq!(liver.conductivity, 0.52);
        assert_eq!(liver.density, 1060.0);
        assert!(liver.has_bioheat_parameters());

        let muscle = tissues::muscle();
        assert_eq!(muscle.conductivity, 0.49);
        assert_eq!(muscle.density, 1090.0);
        assert!(muscle.has_bioheat_parameters());

        let fat = tissues::fat();
        assert_eq!(fat.conductivity, 0.21);
        assert_eq!(fat.density, 911.0);
        assert!(fat.has_bioheat_parameters());

        let tumor = tissues::tumor();
        assert_eq!(tumor.conductivity, 0.55);
        assert_eq!(tumor.density, 1050.0);
        assert!(tumor.has_bioheat_parameters());

        let soft = tissues::soft_tissue();
        assert_eq!(soft.conductivity, 0.5);
        assert_eq!(soft.density, 1050.0);
        assert!(soft.has_bioheat_parameters());
    }

    #[test]
    fn test_thermal_diffusivity() {
        let liver = tissues::liver();
        let alpha = liver.thermal_diffusivity();

        // α = k / (ρc) = 0.52 / (1060 * 3540) ≈ 1.39e-7 m²/s
        let expected = liver.conductivity / (liver.density * liver.specific_heat);
        assert!((alpha - expected).abs() < 1e-12);

        // Should be in reasonable range for tissue (10^-8 to 10^-6 m²/s)
        assert!(alpha > 1e-8 && alpha < 1e-6);
    }

    #[test]
    fn test_bioheat_parameters() {
        let tissue = tissues::liver();

        assert!(tissue.has_bioheat_parameters());
        assert!(tissue.blood_perfusion.is_some());
        assert!(tissue.blood_specific_heat.is_some());

        let w_b = tissue.blood_perfusion.unwrap();
        let c_b = tissue.blood_specific_heat.unwrap();

        assert!(w_b > 0.0);
        assert!(c_b > 0.0);
    }
}
