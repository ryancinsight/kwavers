use super::StateDependentConstants;
use crate::core::constants::water::WaterProperties;

impl StateDependentConstants {
    /// Calculate dynamic viscosity of water with temperature dependence.
    ///
    /// ## Algorithm — Dortmund Data Bank VFT Equation (Vogel-Fulcher-Tammann)
    ///
    /// ### Theorem
    ///
    /// The dynamic viscosity of liquid water follows the three-parameter
    /// Vogel-Fulcher-Tammann (VFT) equation, which generalises the Arrhenius
    /// law η = A·exp(B/T) by introducing a finite singular temperature T₀:
    ///
    /// ```text
    /// η(T) = A · 10^{ B / (T + C) }
    /// ```
    ///
    /// where T is in °C and parameters are fit to NIST experimental data:
    ///
    /// | Parameter | Value | Note |
    /// |-----------|-------|------|
    /// | A | 2.414 × 10⁻⁵ Pa·s | Pre-exponential |
    /// | B | 247.8 °C | Activation term |
    /// | C | 133.15 °C | Vogel shift (T₀ = 140 K) |
    ///
    /// ### Accuracy vs NIST (0–100 °C)
    ///
    /// | T (°C) | Model (mPa·s) | NIST (mPa·s) | Error |
    /// |--------|---------------|--------------|-------|
    /// | 0      | 1.753 | 1.787 | 1.9 % |
    /// | 20     | 1.001 | 1.002 | 0.1 % |
    /// | 37     | 0.690 | 0.692 | 0.3 % |
    /// | 100    | 0.279 | 0.282 | 1.1 % |
    ///
    /// ### References
    ///
    /// 1. Vogel, H. (1921). Das Temperaturabhängigkeitsgesetz der Viskosität von
    ///    Flüssigkeiten. *Physikalische Zeitschrift*, **22**, 645–646.
    ///
    /// 2. Dortmund Data Bank formula: η(T) = 2.414 × 10⁻⁵ × 10^{247.8/(T+133.15)},
    ///    T in °C, η in Pa·s.
    ///
    /// ### Valid range
    ///
    /// 0–100 °C at atmospheric pressure. For other fluids use [`Self::viscosity_arrhenius`].
    ///
    /// # Arguments
    /// * `temperature` — Temperature [°C]
    ///
    /// # Returns
    /// Dynamic viscosity [Pa·s]
    #[must_use]
    pub fn dynamic_viscosity_water(&self, temperature: f64) -> f64 {
        const A: f64 = 2.414e-5; // Pa·s  — pre-exponential factor
        const B: f64 = 247.8; // °C    — activation parameter
        const C: f64 = 133.15; // °C    — Vogel shift (T₀ = 140 K)

        A * 10.0_f64.powf(B / (temperature + C))
    }

    /// Dynamic viscosity using the Arrhenius equation for general fluids.
    ///
    /// ## Theorem — Arrhenius Viscosity Model
    ///
    /// For fluids exhibiting thermally activated flow, the dynamic viscosity
    /// follows:
    ///
    /// ```text
    /// η(T) = A · exp( E_a / (R · T) )
    /// ```
    ///
    /// where:
    ///
    /// | Symbol | Meaning | SI unit |
    /// |--------|---------|---------|
    /// | A | Pre-exponential (∞-T extrapolation) | Pa·s |
    /// | E_a | Molar activation energy for viscous flow | J mol⁻¹ |
    /// | R | Universal gas constant 8.314 J mol⁻¹ K⁻¹ | J mol⁻¹ K⁻¹ |
    /// | T | Absolute temperature | K |
    ///
    /// ## Reference
    ///
    /// Arrhenius, S. (1889). Über die Reaktionsgeschwindigkeit bei der
    /// Inversion von Rohrzucker durch Säuren. *Zeitschrift für physikalische
    /// Chemie*, **4**(1), 226–248.
    ///
    /// # Arguments
    /// * `pre_exponential`    — A [Pa·s]
    /// * `activation_energy`  — E_a [J mol⁻¹]
    /// * `temperature_kelvin` — T (K)
    ///
    /// # Returns
    /// Dynamic viscosity [Pa·s]
    #[must_use]
    pub fn viscosity_arrhenius(
        pre_exponential: f64,
        activation_energy: f64,
        temperature_kelvin: f64,
    ) -> f64 {
        const GAS_CONSTANT: f64 = 8.314_462_618; // J mol⁻¹ K⁻¹ (CODATA 2018)
        pre_exponential * (activation_energy / (GAS_CONSTANT * temperature_kelvin)).exp()
    }

    /// Calculate kinematic viscosity of water: ν = η / ρ
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Kinematic viscosity [m²/s]
    #[must_use]
    pub fn kinematic_viscosity_water(&self, temperature: f64) -> f64 {
        let eta = self.dynamic_viscosity_water(temperature);
        let rho = WaterProperties::density(temperature);
        eta / rho
    }

    /// Calculate thermal diffusivity κ = k/(ρ·Cp)
    ///
    /// Where:
    /// - k = thermal conductivity [W/(m·K)]
    /// - ρ = density [kg/m³]
    /// - Cp = specific heat capacity [J/(kg·K)]
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Thermal diffusivity [m²/s]
    #[must_use]
    pub fn thermal_diffusivity_water(&self, temperature: f64) -> f64 {
        const K_THERM: f64 = 0.598; // W/(m·K) at 20°C
        const CP: f64 = 4182.0; // J/(kg·K)

        let rho = WaterProperties::density(temperature);
        let k_temp = K_THERM * 0.002f64.mul_add(temperature - 20.0, 1.0);

        k_temp / (rho * CP)
    }

    /// Calculate Prandtl number Pr = ν/κ (viscous/thermal diffusivity ratio)
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Prandtl number (dimensionless)
    #[must_use]
    pub fn prandtl_number_water(&self, temperature: f64) -> f64 {
        let nu = self.kinematic_viscosity_water(temperature);
        let kappa = self.thermal_diffusivity_water(temperature);
        nu / kappa
    }

    /// Calculate Reynolds number Re = (ρ·v·L)/η
    ///
    /// # Arguments
    /// * `velocity` - Characteristic velocity (m/s)
    /// * `length` - Characteristic length (m)
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Reynolds number (dimensionless)
    #[must_use]
    pub fn reynolds_number_water(&self, velocity: f64, length: f64, temperature: f64) -> f64 {
        let rho = WaterProperties::density(temperature);
        let eta = self.dynamic_viscosity_water(temperature);

        (rho * velocity * length) / eta
    }
}
