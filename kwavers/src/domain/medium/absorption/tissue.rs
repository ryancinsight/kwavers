//! Tissue-specific absorption models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tissue type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TissueType {
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
pub struct TissueProperties {
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

impl TissueProperties {
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
        let youngs_modulus = 3.0 * bulk_modulus * (1.0 - 2.0 * poisson_ratio);
        let lame_lambda =
            youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
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
            surface_tension: 0.072,    // Similar to water [N/m]
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
pub fn tissue_properties() -> HashMap<TissueType, TissueProperties> {
    let mut map = HashMap::new();

    // Values from literature (Szabo 2004, Duck 1990, HIFU physics literature)
    map.insert(
        TissueType::Water,
        TissueProperties::new(0.0022, 2.0, 1000.0, 1480.0, 5.2, 0.598, 4182.0)
            .with_optical(4.0, 1.0), // Water: μ_a≈0.04 cm⁻¹, μ_s≈0.01 cm⁻¹
    );

    map.insert(
        TissueType::Blood,
        TissueProperties::new(0.18, 1.21, 1060.0, 1575.0, 6.1, 0.52, 3617.0)
            .with_optical(230.0, 20000.0), // Blood: μ_a≈2.3 cm⁻¹, μ_s≈200 cm⁻¹
    );

    map.insert(
        TissueType::Brain,
        TissueProperties::new(0.85, 1.21, 1040.0, 1560.0, 6.6, 0.51, 3630.0)
            .with_optical(20.0, 10000.0), // Brain: μ_a≈0.2 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map.insert(
        TissueType::Fat,
        TissueProperties::new(0.63, 1.1, 950.0, 1450.0, 10.0, 0.21, 2348.0)
            .with_optical(40.0, 15000.0), // Fat: μ_a≈0.4 cm⁻¹, μ_s≈150 cm⁻¹
    );

    map.insert(
        TissueType::Muscle,
        TissueProperties::new(1.3, 1.1, 1090.0, 1580.0, 7.4, 0.49, 3421.0)
            .with_optical(30.0, 10000.0), // Muscle: μ_a≈0.3 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map.insert(
        TissueType::Liver,
        TissueProperties::new(0.94, 1.11, 1060.0, 1570.0, 7.6, 0.52, 3540.0)
            .with_optical(70.0, 10000.0), // Liver: μ_a≈0.7 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map.insert(
        TissueType::Kidney,
        TissueProperties::new(1.0, 1.09, 1050.0, 1560.0, 7.4, 0.53, 3763.0)
            .with_optical(50.0, 12000.0), // Kidney: μ_a≈0.5 cm⁻¹, μ_s≈120 cm⁻¹
    );

    map.insert(
        TissueType::Bone,
        TissueProperties::new(20.0, 1.0, 1900.0, 3500.0, 8.0, 0.32, 1313.0)
            .with_optical(40.0, 35000.0), // Bone: μ_a≈0.4 cm⁻¹, μ_s≈350 cm⁻¹
    );

    map.insert(
        TissueType::Lung,
        TissueProperties::new(41.0, 1.05, 400.0, 650.0, 8.0, 0.39, 3886.0)
            .with_optical(100.0, 25000.0), // Lung: μ_a≈1.0 cm⁻¹, μ_s≈250 cm⁻¹
    );

    map.insert(
        TissueType::Skin,
        TissueProperties::new(2.1, 1.17, 1100.0, 1600.0, 7.5, 0.37, 3391.0)
            .with_optical(50.0, 20000.0), // Skin: μ_a≈0.5 cm⁻¹, μ_s≈200 cm⁻¹
    );

    map.insert(
        TissueType::BreastFat,
        TissueProperties::new(0.75, 1.5, 911.0, 1479.0, 9.6, 0.21, 2348.0)
            .with_optical(40.0, 12000.0), // Breast fat: μ_a≈0.4 cm⁻¹, μ_s≈120 cm⁻¹
    );

    map.insert(
        TissueType::BreastGland,
        TissueProperties::new(0.75, 1.5, 1041.0, 1510.0, 7.0, 0.48, 3600.0)
            .with_optical(60.0, 15000.0), // Breast gland: μ_a≈0.6 cm⁻¹, μ_s≈150 cm⁻¹
    );

    map.insert(
        TissueType::SoftTissue,
        TissueProperties::new(0.75, 1.1, 1050.0, 1540.0, 6.8, 0.50, 3500.0)
            .with_optical(10.0, 10000.0), // Generic soft tissue: μ_a≈0.1 cm⁻¹, μ_s≈100 cm⁻¹
    );

    map
}

/// Static tissue properties accessor
pub static TISSUE_PROPERTIES: std::sync::LazyLock<HashMap<TissueType, TissueProperties>> =
    std::sync::LazyLock::new(tissue_properties);

/// Tissue-specific absorption model
#[derive(Debug, Clone)]
pub struct TissueAbsorption {
    tissue_type: TissueType,
    properties: TissueProperties,
}

impl TissueAbsorption {
    /// Create absorption model for specific tissue
    pub fn new(tissue_type: TissueType) -> Self {
        let properties = TISSUE_PROPERTIES
            .get(&tissue_type)
            .copied()
            .unwrap_or_else(|| TISSUE_PROPERTIES[&TissueType::Water]);

        Self {
            tissue_type,
            properties,
        }
    }

    /// Get tissue properties
    #[must_use]
    pub fn properties(&self) -> &TissueProperties {
        &self.properties
    }

    /// Get tissue type
    #[must_use]
    pub fn tissue_type(&self) -> TissueType {
        self.tissue_type
    }

    /// Calculate absorption coefficient at given frequency
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // Convert to Np/m from dB/(MHz^y cm)
        const DB_TO_NP: f64 = 1.0 / 8.686;
        const CM_TO_M: f64 = 100.0;
        const MHZ_TO_HZ: f64 = 1e6;

        let f_mhz = frequency / MHZ_TO_HZ;
        let alpha_db = self.properties.alpha_0 * f_mhz.powf(self.properties.y);

        alpha_db * DB_TO_NP * CM_TO_M
    }

    /// Get attenuation in dB for given frequency and distance
    #[must_use]
    pub fn attenuation_db(&self, frequency: f64, distance_cm: f64) -> f64 {
        let f_mhz = frequency / 1e6;
        self.properties.alpha_0 * f_mhz.powf(self.properties.y) * distance_cm
    }
}
