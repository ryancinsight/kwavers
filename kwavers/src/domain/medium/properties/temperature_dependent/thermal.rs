use super::super::ThermalPropertyData;

/// Temperature-dependent thermal properties.
///
/// ## Model
/// - Conductivity: `k(T) = k₀[1 + κ₁(T − T₀) + κ₂(T − T₀)²]`
/// - Specific heat: `c_p(T) = c₀[1 + c₁(T − T₀)]`
/// - Diffusivity: `α(T) = k(T) / (ρ(T) c_p(T))`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureDependentThermal {
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
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    /// `k(T) = k₀[1 + κ₁(T − T₀) + κ₂(T − T₀)²]`
    #[inline]
    #[must_use]
    pub fn conductivity(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        let factor = (self.conductivity_coeff_quadratic * delta_t).mul_add(
            delta_t,
            self.conductivity_coeff_linear.mul_add(delta_t, 1.0),
        );
        self.base_properties.conductivity * factor
    }

    /// `c_p(T) = c₀[1 + c₁(T − T₀)]`
    #[inline]
    #[must_use]
    pub fn specific_heat(&self, temperature: f64) -> f64 {
        let delta_t = temperature - self.reference_temperature;
        self.base_properties.specific_heat * self.specific_heat_coefficient.mul_add(delta_t, 1.0)
    }

    /// `α(T) = k(T) / (ρ × c_p(T))`
    #[must_use]
    pub fn thermal_diffusivity(&self, temperature: f64, density: f64) -> f64 {
        self.conductivity(temperature) / (density * self.specific_heat(temperature))
    }

    /// Water properties
    #[must_use]
    pub fn water() -> Self {
        Self {
            base_properties: ThermalPropertyData::water(),
            reference_temperature: 293.15,
            conductivity_coeff_linear: 0.002,
            conductivity_coeff_quadratic: -1e-5,
            specific_heat_coefficient: 0.0001,
        }
    }

    /// Soft tissue properties
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            base_properties: ThermalPropertyData::soft_tissue(),
            reference_temperature: 310.15,
            conductivity_coeff_linear: 0.001,
            conductivity_coeff_quadratic: 0.0,
            specific_heat_coefficient: 0.0002,
        }
    }
}
