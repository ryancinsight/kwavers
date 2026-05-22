use crate::core::constants::thermodynamic::BODY_TEMPERATURE_K;

use super::super::AcousticPropertyData;

/// Temperature-dependent acoustic properties.
///
/// Implements Duck (1990) and Szabo (2004) models for biological tissues.
///
/// ## Models
/// - Sound speed: `c(T) = c₀[1 + β(T − T₀)]` (linear, valid |ΔT| < 20 K)
/// - Density: `ρ(T) = ρ₀[1 − α_T(T − T₀)]` (thermal expansion)
/// - Absorption: `α(T,f) = α₀ f^y [1 + γ(T − T₀)]`
///
/// ## References
/// - Duck, F.A. (1990) "Physical Properties of Tissues", Academic Press.
/// - Bamber, J.C. & Hill, C.R. (1979) Ultrasound Med Biol 5(2):149-157.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureDependentAcoustic {
    pub base_properties: AcousticPropertyData,
    /// Reference temperature T₀ (K)
    pub reference_temperature: f64,
    /// Sound speed temperature coefficient β (K⁻¹)
    pub sound_speed_coefficient: f64,
    /// Volumetric thermal expansion coefficient α_T (K⁻¹)
    pub density_coefficient: f64,
    /// Absorption temperature coefficient γ (K⁻¹)
    pub absorption_coefficient: f64,
}

impl TemperatureDependentAcoustic {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        base_properties: AcousticPropertyData,
        reference_temperature: f64,
        sound_speed_coefficient: f64,
        density_coefficient: f64,
        absorption_coefficient: f64,
    ) -> Result<Self, String> {
        if !(273.0..=373.0).contains(&reference_temperature) {
            return Err(format!(
                "Reference temperature {} K is outside valid range [273, 373] K",
                reference_temperature
            ));
        }
        if sound_speed_coefficient <= 0.0 || sound_speed_coefficient > 0.01 {
            return Err(format!(
                "Sound speed coefficient {} K⁻¹ is outside valid range (0, 0.01] K⁻¹",
                sound_speed_coefficient
            ));
        }
        if density_coefficient <= 0.0 || density_coefficient > 0.001 {
            return Err(format!(
                "Density coefficient {} K⁻¹ is outside valid range (0, 0.001] K⁻¹",
                density_coefficient
            ));
        }
        if !(0.0..0.1).contains(&absorption_coefficient) {
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

    /// `c(T) = c₀[1 + β(T − T₀)]`
    #[inline]
    #[must_use]
    pub fn sound_speed(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        self.base_properties.sound_speed * self.sound_speed_coefficient.mul_add(delta_t, 1.0)
    }

    /// `ρ(T) = ρ₀[1 − α_T(T − T₀)]`
    #[inline]
    #[must_use]
    pub fn density(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        self.base_properties.density * self.density_coefficient.mul_add(-delta_t, 1.0)
    }

    /// `Z(T) = ρ(T) × c(T)`
    #[inline]
    #[must_use]
    pub fn impedance(&self, temperature: f64) -> f64 {
        self.density(temperature) * self.sound_speed(temperature)
    }

    /// `α(T,f) = α₀(T₀) f^y [1 + γ(T − T₀)]`
    #[inline]
    #[must_use]
    pub fn absorption(&self, temperature: f64, frequency: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        self.base_properties.absorption_at_frequency(frequency)
            * self.absorption_coefficient.mul_add(delta_t, 1.0)
    }

    /// Water properties — Duck (1990) Table 2.1
    #[must_use]
    pub fn water() -> Self {
        Self {
            base_properties: AcousticPropertyData::water(),
            reference_temperature: 293.15,
            sound_speed_coefficient: 0.0020,
            density_coefficient: 2.1e-4,
            absorption_coefficient: 0.02,
        }
    }

    /// Soft tissue — Duck (1990) Tables 4.1-4.3
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            base_properties: AcousticPropertyData::soft_tissue(),
            reference_temperature: BODY_TEMPERATURE_K,
            sound_speed_coefficient: 0.0016,
            density_coefficient: 3.7e-4,
            absorption_coefficient: 0.015,
        }
    }

    /// Liver — Bamber & Hill (1979), Duck (1990)
    #[must_use]
    pub fn liver() -> Self {
        Self {
            base_properties: AcousticPropertyData::liver(),
            reference_temperature: BODY_TEMPERATURE_K,
            sound_speed_coefficient: 0.0018,
            density_coefficient: 3.5e-4,
            absorption_coefficient: 0.018,
        }
    }

    /// Muscle — Duck (1990) Table 4.2
    #[must_use]
    pub fn muscle() -> Self {
        Self {
            base_properties: AcousticPropertyData::muscle(),
            reference_temperature: BODY_TEMPERATURE_K,
            sound_speed_coefficient: 0.0016,
            density_coefficient: 3.8e-4,
            absorption_coefficient: 0.012,
        }
    }

    /// Fat — Duck (1990) Table 4.2
    #[must_use]
    pub fn fat() -> Self {
        Self {
            base_properties: AcousticPropertyData::fat(),
            reference_temperature: BODY_TEMPERATURE_K,
            sound_speed_coefficient: 0.0014,
            density_coefficient: 7.0e-4,
            absorption_coefficient: 0.020,
        }
    }
}
