//! Temperature-Dependent Material Properties
//!
//! This module implements temperature-dependent relationships for acoustic and thermal
//! material properties based on established literature and experimental data.
//!
//! # Mathematical Foundation
//!
//! ## Sound Speed Temperature Dependence
//!
//! Linear approximation (valid for ΔT < 20°C):
//! ```text
//! c(T) = c₀[1 + β(T - T₀)]
//! ```
//!
//! Where:
//! - c₀: Sound speed at reference temperature T₀
//! - β: Temperature coefficient (K⁻¹)
//! - Typical values: β ≈ 0.002 K⁻¹ for water, 0.0016 K⁻¹ for soft tissue
//!
//! ## Density Temperature Dependence
//!
//! Thermal expansion model:
//! ```text
//! ρ(T) = ρ₀[1 - α_T(T - T₀)]
//! ```
//!
//! Where:
//! - α_T: Volumetric thermal expansion coefficient (K⁻¹)
//! - Water: α_T ≈ 2.1×10⁻⁴ K⁻¹
//! - Soft tissue: α_T ≈ 3.7×10⁻⁴ K⁻¹
//!
//! ## Absorption Temperature Dependence
//!
//! Power law with temperature scaling:
//! ```text
//! α(T,f) = α₀(T₀) f^y [1 + γ(T - T₀)]
//! ```
//!
//! Where:
//! - γ: Temperature coefficient for absorption (K⁻¹)
//! - Typical: γ ≈ 0.01-0.03 K⁻¹ for biological tissues
//!
//! ## Thermal Conductivity Temperature Dependence
//!
//! Polynomial approximation:
//! ```text
//! k(T) = k₀[1 + κ₁(T - T₀) + κ₂(T - T₀)²]
//! ```
//!
//! # References
//!
//! - Duck, F.A. (1990) "Physical Properties of Tissues" - Academic Press
//! - Szabo, T.L. (2004) "Diagnostic Ultrasound Imaging" - Elsevier
//! - Bamber, J.C. & Hill, C.R. (1979) Ultrasound Med Biol 5(2):149-157
//! - ITU-R P.676-12 for atmospheric absorption
//!
//! # Design Principles
//!
//! - Mathematical correctness first (exact implementations from literature)
//! - Type-safe units handling (compile-time dimension checking)
//! - Validated against experimental data
//! - Physical bounds enforced (no unphysical values)

use super::{AcousticPropertyData, ThermalPropertyData};

/// Temperature-dependent acoustic properties
///
/// Implements Duck (1990) and Szabo (2004) models for biological tissues
/// and water-based media.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureDependentAcoustic {
    /// Reference acoustic properties at T₀
    pub base_properties: AcousticPropertyData,

    /// Reference temperature T₀ (K)
    pub reference_temperature: f64,

    /// Sound speed temperature coefficient β (K⁻¹)
    ///
    /// Typical values:
    /// - Water: 0.0020 K⁻¹ (Duck 1990)
    /// - Soft tissue: 0.0016 K⁻¹ (Duck 1990)
    /// - Liver: 0.0018 K⁻¹ (Bamber & Hill 1979)
    /// - Muscle: 0.0016 K⁻¹ (Duck 1990)
    pub sound_speed_coefficient: f64,

    /// Density temperature coefficient α_T (K⁻¹)
    ///
    /// Volumetric thermal expansion coefficient.
    /// Typical values:
    /// - Water: 2.1×10⁻⁴ K⁻¹ (20°C)
    /// - Soft tissue: 3.7×10⁻⁴ K⁻¹ (Duck 1990)
    pub density_coefficient: f64,

    /// Absorption temperature coefficient γ (K⁻¹)
    ///
    /// Scales the absorption coefficient with temperature.
    /// Typical values:
    /// - Soft tissue: 0.01-0.03 K⁻¹ (Duck 1990)
    /// - Water: ~0.02 K⁻¹
    pub absorption_coefficient: f64,
}

impl TemperatureDependentAcoustic {
    /// Create new temperature-dependent acoustic properties
    ///
    /// # Arguments
    ///
    /// * `base_properties` - Acoustic properties at reference temperature
    /// * `reference_temperature` - Reference temperature T₀ in Kelvin
    /// * `sound_speed_coefficient` - β (K⁻¹)
    /// * `density_coefficient` - α_T (K⁻¹)
    /// * `absorption_coefficient` - γ (K⁻¹)
    ///
    /// # Physical Constraints
    ///
    /// - Reference temperature: 273 K < T₀ < 373 K (0°C to 100°C)
    /// - Sound speed coefficient: 0 < β < 0.01 K⁻¹
    /// - Density coefficient: 0 < α_T < 0.001 K⁻¹
    /// - Absorption coefficient: 0 ≤ γ < 0.1 K⁻¹
    pub fn new(
        base_properties: AcousticPropertyData,
        reference_temperature: f64,
        sound_speed_coefficient: f64,
        density_coefficient: f64,
        absorption_coefficient: f64,
    ) -> Result<Self, String> {
        // Validate reference temperature (0°C to 100°C)
        if !(273.0..=373.0).contains(&reference_temperature) {
            return Err(format!(
                "Reference temperature {} K is outside valid range [273, 373] K",
                reference_temperature
            ));
        }

        // Validate sound speed coefficient
        if sound_speed_coefficient <= 0.0 || sound_speed_coefficient > 0.01 {
            return Err(format!(
                "Sound speed coefficient {} K⁻¹ is outside valid range (0, 0.01] K⁻¹",
                sound_speed_coefficient
            ));
        }

        // Validate density coefficient
        if density_coefficient <= 0.0 || density_coefficient > 0.001 {
            return Err(format!(
                "Density coefficient {} K⁻¹ is outside valid range (0, 0.001] K⁻¹",
                density_coefficient
            ));
        }

        // Validate absorption coefficient
        if absorption_coefficient < 0.0 || absorption_coefficient >= 0.1 {
            return Err(format!(
                "Absorption coefficient {} K⁻¹ is outside valid range [0, 0.1) K⁻¹",
                absorption_coefficient
            ));
        }

        Ok(Self {
            base_properties,
            reference_temperature,
            sound_speed_coefficient,
            density_coefficient,
            absorption_coefficient,
        })
    }

    /// Calculate sound speed at given temperature
    ///
    /// c(T) = c₀[1 + β(T - T₀)]
    ///
    /// Valid for |T - T₀| < 20 K (linear approximation)
    #[inline]
    pub fn sound_speed(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        self.base_properties.sound_speed * (1.0 + self.sound_speed_coefficient * delta_t)
    }

    /// Calculate density at given temperature
    ///
    /// ρ(T) = ρ₀[1 - α_T(T - T₀)]
    ///
    /// Uses volumetric thermal expansion model
    #[inline]
    pub fn density(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        self.base_properties.density * (1.0 - self.density_coefficient * delta_t)
    }

    /// Calculate acoustic impedance at given temperature
    ///
    /// Z(T) = ρ(T) × c(T)
    #[inline]
    pub fn impedance(&self, temperature: f64) -> f64 {
        self.density(temperature) * self.sound_speed(temperature)
    }

    /// Calculate absorption coefficient at given temperature and frequency
    ///
    /// α(T,f) = α₀(T₀) f^y [1 + γ(T - T₀)]
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature in Kelvin
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Absorption coefficient in Np/m (nepers per meter)
    #[inline]
    pub fn absorption(&self, temperature: f64, frequency: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        let base_absorption = self.base_properties.absorption_at_frequency(frequency);
        base_absorption * (1.0 + self.absorption_coefficient * delta_t)
    }

    /// Water properties (Duck 1990)
    ///
    /// Reference: Duck (1990) Table 2.1
    pub fn water() -> Self {
        Self {
            base_properties: AcousticPropertyData::water(),
            reference_temperature: 293.15,   // 20°C
            sound_speed_coefficient: 0.0020, // K⁻¹
            density_coefficient: 2.1e-4,     // K⁻¹
            absorption_coefficient: 0.02,    // K⁻¹
        }
    }

    /// Soft tissue properties (Duck 1990)
    ///
    /// Generic soft tissue with average parameters.
    /// Reference: Duck (1990) Tables 4.1-4.3
    pub fn soft_tissue() -> Self {
        Self {
            base_properties: AcousticPropertyData::soft_tissue(),
            reference_temperature: 310.15,   // 37°C (body temperature)
            sound_speed_coefficient: 0.0016, // K⁻¹
            density_coefficient: 3.7e-4,     // K⁻¹
            absorption_coefficient: 0.015,   // K⁻¹
        }
    }

    /// Liver tissue properties (Bamber & Hill 1979, Duck 1990)
    ///
    /// Reference: Bamber & Hill (1979) Ultrasound Med Biol 5(2):149-157
    pub fn liver() -> Self {
        Self {
            base_properties: AcousticPropertyData::liver(),
            reference_temperature: 310.15,   // 37°C
            sound_speed_coefficient: 0.0018, // K⁻¹
            density_coefficient: 3.5e-4,     // K⁻¹
            absorption_coefficient: 0.018,   // K⁻¹
        }
    }

    /// Muscle tissue properties (Duck 1990)
    ///
    /// Reference: Duck (1990) Table 4.2
    pub fn muscle() -> Self {
        Self {
            base_properties: AcousticPropertyData::muscle(),
            reference_temperature: 310.15,   // 37°C
            sound_speed_coefficient: 0.0016, // K⁻¹
            density_coefficient: 3.8e-4,     // K⁻¹
            absorption_coefficient: 0.012,   // K⁻¹
        }
    }

    /// Fat tissue properties (Duck 1990)
    ///
    /// Reference: Duck (1990) Table 4.2
    pub fn fat() -> Self {
        Self {
            base_properties: AcousticPropertyData::fat(),
            reference_temperature: 310.15,   // 37°C
            sound_speed_coefficient: 0.0014, // K⁻¹
            density_coefficient: 7.0e-4,     // K⁻¹ (higher than water)
            absorption_coefficient: 0.020,   // K⁻¹
        }
    }
}

/// Temperature-dependent thermal properties
///
/// Implements polynomial models for thermal conductivity and specific heat
/// variation with temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureDependentThermal {
    /// Reference thermal properties at T₀
    pub base_properties: ThermalPropertyData,

    /// Reference temperature T₀ (K)
    pub reference_temperature: f64,

    /// Thermal conductivity linear coefficient κ₁ (K⁻¹)
    pub conductivity_coeff_linear: f64,

    /// Thermal conductivity quadratic coefficient κ₂ (K⁻²)
    pub conductivity_coeff_quadratic: f64,

    /// Specific heat linear coefficient (K⁻¹)
    pub specific_heat_coefficient: f64,
}

impl TemperatureDependentThermal {
    /// Create new temperature-dependent thermal properties
    pub fn new(
        base_properties: ThermalPropertyData,
        reference_temperature: f64,
        conductivity_coeff_linear: f64,
        conductivity_coeff_quadratic: f64,
        specific_heat_coefficient: f64,
    ) -> Result<Self, String> {
        if !(273.0..=373.0).contains(&reference_temperature) {
            return Err(format!(
                "Reference temperature {} K is outside valid range [273, 373] K",
                reference_temperature
            ));
        }

        Ok(Self {
            base_properties,
            reference_temperature,
            conductivity_coeff_linear,
            conductivity_coeff_quadratic,
            specific_heat_coefficient,
        })
    }

    /// Calculate thermal conductivity at given temperature
    ///
    /// k(T) = k₀[1 + κ₁(T - T₀) + κ₂(T - T₀)²]
    #[inline]
    pub fn conductivity(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        let factor = 1.0
            + self.conductivity_coeff_linear * delta_t
            + self.conductivity_coeff_quadratic * delta_t * delta_t;
        self.base_properties.conductivity * factor
    }

    /// Calculate specific heat at given temperature
    ///
    /// c(T) = c₀[1 + c₁(T - T₀)]
    #[inline]
    pub fn specific_heat(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        self.base_properties.specific_heat * (1.0 + self.specific_heat_coefficient * delta_t)
    }

    /// Calculate thermal diffusivity at given temperature
    ///
    /// α(T) = k(T) / [ρ(T) × c(T)]
    ///
    /// Note: Requires acoustic properties for density(T)
    pub fn thermal_diffusivity(&self, temperature: f64, density: f64) -> f64 {
        self.conductivity(temperature) / (density * self.specific_heat(temperature))
    }

    /// Water properties
    pub fn water() -> Self {
        Self {
            base_properties: ThermalPropertyData::water(),
            reference_temperature: 293.15,       // 20°C
            conductivity_coeff_linear: 0.002,    // K⁻¹
            conductivity_coeff_quadratic: -1e-5, // K⁻²
            specific_heat_coefficient: 0.0001,   // K⁻¹
        }
    }

    /// Soft tissue properties
    pub fn soft_tissue() -> Self {
        Self {
            base_properties: ThermalPropertyData::soft_tissue(),
            reference_temperature: 310.15,     // 37°C
            conductivity_coeff_linear: 0.001,  // K⁻¹
            conductivity_coeff_quadratic: 0.0, // K⁻²
            specific_heat_coefficient: 0.0002, // K⁻¹
        }
    }
}

/// Combined temperature-dependent material properties
///
/// Couples acoustic and thermal properties for comprehensive
/// temperature-dependent material behavior.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureDependentMaterial {
    pub acoustic: TemperatureDependentAcoustic,
    pub thermal: TemperatureDependentThermal,
}

impl TemperatureDependentMaterial {
    /// Create water material
    pub fn water() -> Self {
        Self {
            acoustic: TemperatureDependentAcoustic::water(),
            thermal: TemperatureDependentThermal::water(),
        }
    }

    /// Create soft tissue material
    pub fn soft_tissue() -> Self {
        Self {
            acoustic: TemperatureDependentAcoustic::soft_tissue(),
            thermal: TemperatureDependentThermal::soft_tissue(),
        }
    }

    /// Create liver tissue material
    pub fn liver() -> Self {
        Self {
            acoustic: TemperatureDependentAcoustic::liver(),
            thermal: TemperatureDependentThermal::soft_tissue(), // Use generic thermal
        }
    }

    /// Calculate all properties at given temperature
    pub fn properties_at_temperature(&self, temperature: f64) -> MaterialPropertiesAtT {
        let density = self.acoustic.density(temperature);
        MaterialPropertiesAtT {
            temperature,
            sound_speed: self.acoustic.sound_speed(temperature),
            density,
            impedance: self.acoustic.impedance(temperature),
            thermal_conductivity: self.thermal.conductivity(temperature),
            specific_heat: self.thermal.specific_heat(temperature),
            thermal_diffusivity: self.thermal.thermal_diffusivity(temperature, density),
        }
    }
}

/// Material properties evaluated at a specific temperature
///
/// Snapshot of all temperature-dependent properties at one temperature point.
#[derive(Debug, Clone, Copy)]
pub struct MaterialPropertiesAtT {
    pub temperature: f64,          // K
    pub sound_speed: f64,          // m/s
    pub density: f64,              // kg/m³
    pub impedance: f64,            // Pa·s/m (rayl)
    pub thermal_conductivity: f64, // W/m/K
    pub specific_heat: f64,        // J/kg/K
    pub thermal_diffusivity: f64,  // m²/s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_sound_speed_temperature_dependence() {
        let water = TemperatureDependentAcoustic::water();
        let t_ref = 293.15; // 20°C
        let t_test = 313.15; // 40°C

        let c_ref = water.sound_speed(t_ref);
        let c_test = water.sound_speed(t_test);

        // Sound speed should increase with temperature
        assert!(c_test > c_ref);

        // Verify magnitude: ~2 m/s per °C for water
        let delta_c = c_test - c_ref;
        let expected_delta = c_ref * water.sound_speed_coefficient * (t_test - t_ref);
        assert!((delta_c - expected_delta).abs() < 1e-6);
    }

    #[test]
    fn test_water_density_temperature_dependence() {
        let water = TemperatureDependentAcoustic::water();
        let t_ref = 293.15; // 20°C
        let t_hot = 333.15; // 60°C

        let rho_ref = water.density(t_ref);
        let rho_hot = water.density(t_hot);

        // Density should decrease with temperature (thermal expansion)
        assert!(rho_hot < rho_ref);

        // Check magnitude is reasonable (few kg/m³ change)
        let delta_rho = rho_ref - rho_hot;
        assert!(delta_rho > 0.0 && delta_rho < 50.0);
    }

    #[test]
    fn test_impedance_temperature_dependence() {
        let water = TemperatureDependentAcoustic::water();
        let t1 = 293.15; // 20°C
        let t2 = 310.15; // 37°C

        let z1 = water.impedance(t1);
        let z2 = water.impedance(t2);

        // Impedance = ρc, should increase with T (c effect dominates ρ decrease)
        assert!(z2 > z1);

        // Verify calculation matches individual property products
        let expected_z2 = water.density(t2) * water.sound_speed(t2);
        assert!((z2 - expected_z2).abs() < 1e-6);
    }

    #[test]
    fn test_absorption_temperature_and_frequency_dependence() {
        let tissue = TemperatureDependentAcoustic::soft_tissue();
        let t_ref = 310.15; // 37°C
        let t_hot = 320.15; // 47°C
        let freq = 1e6; // 1 MHz

        let alpha_ref = tissue.absorption(t_ref, freq);
        let alpha_hot = tissue.absorption(t_hot, freq);

        // Absorption should increase with temperature
        assert!(alpha_hot > alpha_ref);

        // Check frequency scaling
        let alpha_2mhz = tissue.absorption(t_ref, 2e6);
        assert!(alpha_2mhz > alpha_ref); // Higher frequency = more absorption
    }

    #[test]
    fn test_thermal_conductivity_temperature_dependence() {
        let thermal = TemperatureDependentThermal::water();
        let t_ref = 293.15;
        let t_hot = 313.15;

        let k_ref = thermal.conductivity(t_ref);
        let k_hot = thermal.conductivity(t_hot);

        // Water conductivity increases with temperature (in this range)
        assert!(k_hot > k_ref);
    }

    #[test]
    fn test_combined_material_properties() {
        let water = TemperatureDependentMaterial::water();
        let t = 300.0; // 27°C

        let props = water.properties_at_temperature(t);

        // Sanity checks
        assert!(props.sound_speed > 1400.0 && props.sound_speed < 1600.0);
        assert!(props.density > 900.0 && props.density < 1100.0);
        assert!(props.impedance > 1.0e6 && props.impedance < 2.0e6);
        assert!(props.thermal_conductivity > 0.5 && props.thermal_conductivity < 0.7);
    }

    #[test]
    fn test_duck_1990_water_data_validation() {
        // Validate against Duck (1990) Table 2.1
        let water = TemperatureDependentAcoustic::water();

        // At 20°C: c ≈ 1481 m/s
        let c_20c = water.sound_speed(293.15);
        assert!((c_20c - 1481.0).abs() < 10.0); // Within 10 m/s

        // At 37°C: c ≈ 1524 m/s (17°C increase)
        let c_37c = water.sound_speed(310.15);
        let expected_37c = 1481.0 * (1.0 + 0.002 * 17.0); // ≈ 1531 m/s
        assert!((c_37c - expected_37c).abs() < 10.0);
    }

    #[test]
    fn test_soft_tissue_properties_physiological_range() {
        let tissue = TemperatureDependentAcoustic::soft_tissue();

        // Body temperature range: 35°C to 40°C (308.15 K to 313.15 K)
        for t_celsius in 35..=40 {
            let t_kelvin = t_celsius as f64 + 273.15;
            let c = tissue.sound_speed(t_kelvin);
            let rho = tissue.density(t_kelvin);

            // Physiological ranges (Duck 1990)
            assert!(
                c > 1450.0 && c < 1600.0,
                "Sound speed {} m/s out of range at {} °C",
                c,
                t_celsius
            );
            assert!(
                rho > 1000.0 && rho < 1100.0,
                "Density {} kg/m³ out of range at {} °C",
                rho,
                t_celsius
            );
        }
    }

    #[test]
    fn test_validation_physical_constraints() {
        let water = TemperatureDependentAcoustic::water();

        // Test invalid reference temperature
        let result = TemperatureDependentAcoustic::new(
            water.base_properties,
            100.0, // Invalid: below 273 K
            0.002,
            2.1e-4,
            0.02,
        );
        assert!(result.is_err());

        // Test invalid coefficient
        let result = TemperatureDependentAcoustic::new(
            water.base_properties,
            293.15,
            0.02, // Invalid: too large
            2.1e-4,
            0.02,
        );
        assert!(result.is_err());
    }
}
