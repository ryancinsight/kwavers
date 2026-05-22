//! Tissue-specific absorption models

use crate::core::constants::acoustic_parameters::{BONE_DENSITY, BONE_SOUND_SPEED};
use crate::core::constants::cavitation::SURFACE_TENSION_WATER;
use crate::core::constants::fundamental::{
    DENSITY_BLOOD, DENSITY_BRAIN, DENSITY_BREAST_FAT, DENSITY_FAT, DENSITY_LIVER,
    DENSITY_MUSCLE, DENSITY_TISSUE, DENSITY_WATER_NOMINAL, SOUND_SPEED_BLOOD, SOUND_SPEED_BRAIN,
    SOUND_SPEED_KIDNEY, SOUND_SPEED_LIVER, SOUND_SPEED_MUSCLE, SOUND_SPEED_TISSUE,
    SOUND_SPEED_WATER,
};
use crate::core::constants::numerical::{CM_TO_M, MHZ_TO_HZ};
use crate::core::constants::thermodynamic::SPECIFIC_HEAT_WATER;
use crate::core::constants::DB_TO_NP;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tissue type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AbsorptionTissueType {
    Water,
    Blood,
    Brain,
    Fat,
    Muscle,
    Liver,
    Kidney,
    Bone,
    Lung,
    Skin,
    BreastFat,
    BreastGland,
    SoftTissue, // Generic soft tissue
}

/// Tissue absorption properties
#[derive(Debug, Clone, Copy)]
pub struct AbsorptionTissueProperties {
    /// Absorption coefficient at 1 `MHz` [dB/(MHz^y cm)]
    pub alpha_0: f64,
    /// Alternative notation for `alpha_0`
    pub alpha0: f64,
    /// Power law exponent
    pub y: f64,
    /// Alternative notation for power law exponent
    pub delta: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Nonlinearity parameter B/A
    pub nonlinearity: f64,
    /// Alternative notation for B/A
    pub b_a: f64,
    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,
    /// Specific heat capacity [J/(kg·K)]
    pub specific_heat: f64,
    /// Lamé first parameter \[Pa\]
    pub lame_lambda: f64,
    /// Lamé second parameter (shear modulus) \[Pa\]
    pub lame_mu: f64,
    /// Thermal expansion coefficient [1/K]
    pub thermal_expansion: f64,
    /// Dynamic viscosity [Pa·s]
    pub viscosity: f64,
    /// Surface tension [N/m]
    pub surface_tension: f64,
    /// Gas diffusion coefficient [m²/s]
    pub gas_diffusion_coefficient: f64,
    /// Optical absorption coefficient at ~800nm [1/m]
    pub optical_absorption_coeff: f64,
    /// Optical scattering coefficient at ~800nm [1/m]
    pub optical_scattering_coeff: f64,
}

impl AbsorptionTissueProperties {
    /// Create tissue properties with consistent fields
    fn new(
        alpha: f64,
        y: f64,
        density: f64,
        sound_speed: f64,
        nonlinearity: f64,
        thermal: f64,
        heat_cap: f64,
    ) -> Self {
        // Calculate Lamé parameters from sound speed and density
        // For soft tissue, assume Poisson's ratio of 0.499 (nearly incompressible)
        let bulk_modulus = density * sound_speed * sound_speed;
        let poisson_ratio = 0.499;
        let youngs_modulus = 3.0 * bulk_modulus * 2.0f64.mul_add(-poisson_ratio, 1.0);
        let lame_lambda = youngs_modulus * poisson_ratio
            / ((1.0 + poisson_ratio) * 2.0f64.mul_add(-poisson_ratio, 1.0));
        let lame_mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));

        Self {
            alpha_0: alpha,
            alpha0: alpha, // Duplicate for compatibility
            y,
            delta: y, // Duplicate for compatibility
            density,
            sound_speed,
            nonlinearity,
            b_a: nonlinearity, // Duplicate for compatibility
            thermal_conductivity: thermal,
            specific_heat: heat_cap,
            lame_lambda,
            lame_mu,
            thermal_expansion: 3.0e-4, // Typical for soft tissue [1/K]
            viscosity: 0.003,          // Typical for tissue [Pa·s]
            surface_tension: SURFACE_TENSION_WATER, // Similar to water [N/m]
            gas_diffusion_coefficient: 2e-9, // Oxygen in tissue [m²/s]
            // Default optical properties for generic soft tissue at ~800nm (NIR)
            // Reference: Jacques (2013) "Optical properties of biological tissues: a review"
            optical_absorption_coeff: 10.0,    // 0.1 cm⁻¹ = 10 m⁻¹
            optical_scattering_coeff: 10000.0, // 100 cm⁻¹ = 10000 m⁻¹
        }
    }

    /// Set tissue-specific optical properties (absorption & scattering at ~800nm)
    ///
    /// Values in [1/m]. Reference: Jacques (2013), Cheong et al. (1990)
    #[must_use]
    fn with_optical(mut self, absorption: f64, scattering: f64) -> Self {
        self.optical_absorption_coeff = absorption;
        self.optical_scattering_coeff = scattering;
        self
    }
}

/// Get tissue properties database
#[must_use]
pub fn tissue_properties() -> HashMap<AbsorptionTissueType, AbsorptionTissueProperties> {
    let mut map = HashMap::new();

    // Values from literature (Szabo 2004, Duck 1990, HIFU physics literature)
    map.insert(
        AbsorptionTissueType::Water,
        AbsorptionTissueProperties::new(
            0.0022,
            2.0,
            DENSITY_WATER_NOMINAL,
            SOUND_SPEED_WATER,
            5.2,
            0.598,
            SPECIFIC_HEAT_WATER,
        )
        .with_optical(4.0, 1.0), // Water: μ_a≈0.04 cm⁻¹, μ_s≈0.01 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Blood,
        AbsorptionTissueProperties::new(0.18, 1.21, DENSITY_BLOOD, SOUND_SPEED_BLOOD, 6.1, 0.52, 3617.0)
            .with_optical(230.0, 20000.0), // Blood: μ_a≈2.3 cm⁻¹, μ_s≈200 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Brain,
        AbsorptionTissueProperties::new(0.85, 1.21, DENSITY_BRAIN, SOUND_SPEED_BRAIN, 6.6, 0.51, 3630.0)
            .with_optical(20.0, 10000.0), // Brain: μ_a≈0.2 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Fat,
        AbsorptionTissueProperties::new(0.63, 1.1, DENSITY_FAT, 1450.0, 10.0, 0.21, 2348.0)
            .with_optical(40.0, 15000.0), // Fat: μ_a≈0.4 cm⁻¹, μ_s≈150 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Muscle,
        AbsorptionTissueProperties::new(1.3, 1.1, DENSITY_MUSCLE, SOUND_SPEED_MUSCLE, 7.4, 0.49, 3421.0)
            .with_optical(30.0, 10000.0), // Muscle: μ_a≈0.3 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Liver,
        AbsorptionTissueProperties::new(0.94, 1.11, DENSITY_LIVER, SOUND_SPEED_LIVER, 7.6, 0.52, 3540.0)
            .with_optical(70.0, 10000.0), // Liver: μ_a≈0.7 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Kidney,
        AbsorptionTissueProperties::new(1.0, 1.09, DENSITY_TISSUE, SOUND_SPEED_KIDNEY, 7.4, 0.53, 3763.0)
            .with_optical(50.0, 12000.0), // Kidney: μ_a≈0.5 cm⁻¹, μ_s≈120 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Bone,
        AbsorptionTissueProperties::new(20.0, 1.0, BONE_DENSITY, BONE_SOUND_SPEED, 8.0, 0.32, 1313.0)
            .with_optical(40.0, 35000.0), // Bone: μ_a≈0.4 cm⁻¹, μ_s≈350 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Lung,
        AbsorptionTissueProperties::new(41.0, 1.05, 400.0, 650.0, 8.0, 0.39, 3886.0)
            .with_optical(100.0, 25000.0), // Lung: μ_a≈1.0 cm⁻¹, μ_s≈250 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::Skin,
        AbsorptionTissueProperties::new(2.1, 1.17, 1100.0, 1600.0, 7.5, 0.37, 3391.0)
            .with_optical(50.0, 20000.0), // Skin: μ_a≈0.5 cm⁻¹, μ_s≈200 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::BreastFat,
        AbsorptionTissueProperties::new(0.75, 1.5, DENSITY_BREAST_FAT, 1479.0, 9.6, 0.21, 2348.0)
            .with_optical(40.0, 12000.0), // Breast fat: μ_a≈0.4 cm⁻¹, μ_s≈120 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::BreastGland,
        AbsorptionTissueProperties::new(0.75, 1.5, 1041.0, 1510.0, 7.0, 0.48, 3600.0)
            .with_optical(60.0, 15000.0), // Breast gland: μ_a≈0.6 cm⁻¹, μ_s≈150 cm⁻¹
    );

    map.insert(
        AbsorptionTissueType::SoftTissue,
        AbsorptionTissueProperties::new(0.75, 1.1, DENSITY_TISSUE, SOUND_SPEED_TISSUE, 6.8, 0.50, 3500.0)
            .with_optical(10.0, 10000.0), // Generic soft tissue: μ_a≈0.1 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map
}

/// Static tissue properties accessor
pub static TISSUE_PROPERTIES: std::sync::LazyLock<
    HashMap<AbsorptionTissueType, AbsorptionTissueProperties>,
> = std::sync::LazyLock::new(tissue_properties);

/// Tissue-specific absorption model
#[derive(Debug, Clone)]
pub struct TissueAbsorption {
    tissue_type: AbsorptionTissueType,
    properties: AbsorptionTissueProperties,
}

impl TissueAbsorption {
    /// Create absorption model for specific tissue
    pub fn new(tissue_type: AbsorptionTissueType) -> Self {
        let properties = TISSUE_PROPERTIES
            .get(&tissue_type)
            .copied()
            .unwrap_or_else(|| TISSUE_PROPERTIES[&AbsorptionTissueType::Water]);

        Self {
            tissue_type,
            properties,
        }
    }

    /// Get tissue properties
    #[must_use]
    pub fn properties(&self) -> &AbsorptionTissueProperties {
        &self.properties
    }

    /// Get tissue type
    #[must_use]
    pub fn tissue_type(&self) -> AbsorptionTissueType {
        self.tissue_type
    }

    /// Calculate absorption coefficient at given frequency
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // Convert dB/(MHz^y cm) → Np/m:
        //   α [Np/m] = α [dB/cm] · (1/CM_TO_M) [cm/m] · DB_TO_NP [Np/dB]
        let f_mhz = frequency / MHZ_TO_HZ;
        let alpha_db = self.properties.alpha_0 * f_mhz.powf(self.properties.y);

        alpha_db * DB_TO_NP / CM_TO_M
    }

    /// Get attenuation in dB for given frequency and distance
    #[must_use]
    pub fn attenuation_db(&self, frequency: f64, distance_cm: f64) -> f64 {
        let f_mhz = frequency / MHZ_TO_HZ;
        self.properties.alpha_0 * f_mhz.powf(self.properties.y) * distance_cm
    }
}
