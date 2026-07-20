//! Temperature-Dependent Material Properties
//!
//! ## Mathematical Foundation
//!
//! Sound speed: `c(T) = c₀[1 + β(T − T₀)]` (Duck 1990)
//! Density: `ρ(T) = ρ₀[1 − α_T(T − T₀)]`
//! Absorption: `α(T,f) = α₀ f^y [1 + γ(T − T₀)]`
//! Conductivity: `k(T) = k₀[1 + κ₁(T − T₀) + κ₂(T − T₀)²]`
//!
//! ## References
//! - Duck, F.A. (1990) "Physical Properties of Tissues", Academic Press.
//! - Szabo, T.L. (2004) "Diagnostic Ultrasound Imaging", Elsevier.
//! - Bamber, J.C. & Hill, C.R. (1979) Ultrasound Med Biol 5(2):149-157.

mod acoustic;
#[cfg(test)]
mod tests;
mod thermal;

pub use acoustic::TemperatureDependentAcoustic;
pub use thermal::TemperatureDependentThermal;

/// Combined temperature-dependent material (acoustic + thermal).
#[derive(Debug, Clone, Copy)]
pub struct TemperatureDependentMaterial {
    pub acoustic: TemperatureDependentAcoustic,
    pub thermal: TemperatureDependentThermal,
}

impl TemperatureDependentMaterial {
    #[must_use]
    pub fn water() -> Self {
        Self {
            acoustic: TemperatureDependentAcoustic::water(),
            thermal: TemperatureDependentThermal::water(),
        }
    }

    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            acoustic: TemperatureDependentAcoustic::soft_tissue(),
            thermal: TemperatureDependentThermal::soft_tissue(),
        }
    }

    #[must_use]
    pub fn liver() -> Self {
        Self {
            acoustic: TemperatureDependentAcoustic::liver(),
            thermal: TemperatureDependentThermal::soft_tissue(),
        }
    }

    /// Evaluate every property at `temperature` kelvin.
    ///
    /// # Errors
    ///
    /// Returns an error when the thermal constitutive law rejects the supplied
    /// temperature or its derived properties.
    pub fn properties_at_temperature(
        &self,
        temperature: f64,
    ) -> Result<MaterialPropertiesAtT, String> {
        let density = self.acoustic.density(temperature);
        let thermal = self.thermal.properties_with_density(temperature, density)?;
        Ok(MaterialPropertiesAtT {
            temperature,
            sound_speed: self.acoustic.sound_speed(temperature),
            density,
            impedance: self.acoustic.impedance(temperature),
            thermal_conductivity: thermal.conductivity(),
            specific_heat: thermal.specific_heat(),
            thermal_diffusivity: thermal.thermal_diffusivity(),
        })
    }
}

/// All temperature-dependent properties evaluated at one temperature point.
#[derive(Debug, Clone, Copy)]
pub struct MaterialPropertiesAtT {
    pub temperature: f64,
    pub sound_speed: f64,
    pub density: f64,
    pub impedance: f64,
    pub thermal_conductivity: f64,
    pub specific_heat: f64,
    pub thermal_diffusivity: f64,
}
