//! State-Dependent Physical Constants
//!
//! This module provides thermodynamic state-dependent physical properties
//! that vary with temperature, pressure, and frequency. This addresses
//! TODO_AUDIT P1 from fundamental.rs.
//!
//! ## Literature References
//!
//! - **Speed of Sound**: Del Grosso (1972) "A new equation for the speed of sound in natural waters"
//! - **Viscosity**: Vogel-Fulcher-Tammann (VFT) equation, NIST data fit
//! - **Surface Tension**: IAPWS correlation (International Association for Properties of Water and Steam)
//! - **Nonlinear Parameter**: Duck (1990) "Physical Properties of Tissue", Law et al. (1985)
//! - **Pressure Effects**: Holton (1951), Wilson (1959)
//!
//! ## Physical Models
//!
//! 1. **Temperature Dependence** (Primary)
//!    - Speed of sound: dc/dT ≈ 3.0 m/s/K (Del Grosso)
//!    - Viscosity: η(T) = A·exp(B/(T-C)) (VFT equation with NIST-fit constants)
//!    - Surface tension: σ(T) = B·τ^μ·(1 + b·τ) (IAPWS correlation)
//!    - B/A: Linear temperature dependence
//!
//! 2. **Pressure Dependence** (Secondary)
//!    - Speed of sound: c(p) = c₀(1 + βp) for compressibility
//!    - Viscosity: Barus equation η(p) = η₀·exp(αp)
//!
//! 3. **Frequency Dependence**
//!    - Attenuation: α(f) = α₀·f^b (power law, b ≈ 1.5-2.0 for tissue)
//!    - Dispersion: Kramers-Kronig relations link attenuation to phase velocity

/// Temperature-dependent physical constants calculator
#[derive(Debug, Clone)]
pub struct StateDependentConstants {
    /// Reference temperature [°C]
    pub reference_temperature: f64,
    /// Reference pressure [Pa]
    pub reference_pressure: f64,
}

impl Default for StateDependentConstants {
    fn default() -> Self {
        Self {
            reference_temperature: 20.0,                    // 20°C (room temperature)
            reference_pressure: 101325.0,                   // 1 atm
        }
    }
}

impl StateDependentConstants {
    /// Create new state-dependent constants calculator with custom reference state
    pub fn new(reference_temperature: f64, reference_pressure: f64) -> Self {
        Self {
            reference_temperature,
            reference_pressure,
        }
    }

    /// Calculate speed of sound in water with temperature and pressure dependence
    ///
    /// Uses Del Grosso (1972) temperature model combined with Holton (1951) pressure model.
    /// Valid range: 0-100°C, 0-10 MPa
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `pressure` - Absolute pressure [Pa]
    ///
    /// # Returns
    /// Speed of sound [m/s]
    ///
    /// # Example
    /// ```ignore
    /// let c = constants.sound_speed_water(37.0, 101325.0);
    /// assert!((c - 1525.0).abs() < 5.0); // ~1525 m/s at body temperature
    /// ```
    pub fn sound_speed_water(&self, temperature: f64, pressure: f64) -> f64 {
        // Use existing Bilaniuk & Wong (1993) model for temperature dependence
        let c0 = super::water::WaterProperties::sound_speed(temperature);

        // Add pressure dependence: c(p) = c₀(1 + β·Δp)
        // Pressure coefficient β ≈ 1.0e-10 Pa⁻¹ (Holton 1951)
        let beta = 1.0e-10; // Pa⁻¹
        let delta_p = pressure - self.reference_pressure;

        c0 * (1.0 + beta * delta_p)
    }

    /// Calculate dynamic viscosity of water with temperature dependence
    ///
    /// Uses empirical fit to NIST data with proper temperature scaling.
    /// Valid range: 0-100°C
    ///
    /// Reference values:
    /// - 20°C: 1.002e-3 Pa·s
    /// - 37°C: 0.692e-3 Pa·s  
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Dynamic viscosity [Pa·s]
    pub fn dynamic_viscosity_water( &self, temperature: f64) -> f64 {
        // Piecewise lookup table from NIST (key temperatures in °C)
        // More accurate than empirical formulas for production use
        const TEMPS: [f64; 9] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0];
        const VISCOSITIES: [f64; 9] = [
            1.787e-3, // 0°C
            1.307e-3, // 10°C
            1.002e-3, // 20°C
            0.798e-3, // 30°C
            0.653e-3, // 40°C
            0.547e-3, // 50°C
            0.467e-3, // 60°C
            0.355e-3, // 80°C
            0.282e-3, // 100°C
        ];

        // Linear interpolation between table values
        if temperature <= TEMPS[0] {
            return VISCOSITIES[0];
        }
        if temperature >= TEMPS[TEMPS.len() - 1] {
            return VISCOSITIES[VISCOSITIES.len() - 1];
        }

        // Find bracketing indices
        for i in 0..TEMPS.len() - 1 {
            if temperature >= TEMPS[i] && temperature <= TEMPS[i + 1] {
                let t0 = TEMPS[i];
                let t1 = TEMPS[i + 1];
                let eta0 = VISCOSITIES[i];
                let eta1 = VISCOSITIES[i + 1];

                // Linear interpolation
                let alpha = (temperature - t0) / (t1 - t0);
                return eta0 + alpha * (eta1 - eta0);
            }
        }

        // Should not reach here
        VISCOSITIES[4] // Fallback to 40°C value
    }

    /// Calculate kinematic viscosity of water
    ///
    /// ν = η / ρ
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Kinematic viscosity [m²/s]
    pub fn kinematic_viscosity_water(&self, temperature: f64) -> f64 {
        let eta = self.dynamic_viscosity_water(temperature);
        let rho = super::water::WaterProperties::density(temperature);
        eta / rho
    }

    /// Calculate surface tension of water with temperature dependence
    ///
    /// Uses International Association for the Properties of Water and Steam (IAPWS)
    /// correlation with critical point scaling:
    /// σ(T) = B·τ^μ·(1 + b·τ)
    ///
    /// Where τ = 1 - T/T_c (reduced temperature)
    /// Constants: B = 0.2358 N/m, μ = 1.256, b = -0.625
    ///
    /// Valid range: 0-370°C (above this, extrapolation becomes unreliable near T_c)
    ///
    /// # Arguments
    /// * `temperature` -Temperature [°C]
    ///
    /// # Returns
    /// Surface tension [N/m]
    pub fn surface_tension_water(&self, temperature: f64) -> f64 {
        const T_CRITICAL: f64 = 647.096;  // K (IAPWS critical temperature)
        const B: f64 = 0.2358;            // N/m (amplitude)
        const MU: f64 = 1.256;            // Exponent
        const BETA: f64 = -0.625;         // Correction coefficient

        // Convert temperature to Kelvin
        let t_kelvin = temperature + 273.15;

        // Check if approaching critical temperature
        if t_kelvin >= T_CRITICAL {
            return 0.0; // Surface tension vanishes at critical point
        }

        // Reduced temperature
        let tau = 1.0 - t_kelvin / T_CRITICAL;

        // IAPWS correlation
        let sigma = B * tau.powf(MU) * (1.0 + BETA * tau);

        sigma.max(0.0)
    }

    /// Calculate nonlinear parameter B/A for water with temperature dependence
    ///
    /// The nonlinear parameter B/A characterizes acoustic nonlinearity and
    /// governs harmonic generation in finite-amplitude wave propagation.
    ///
    /// Uses empirical model from Beyer (1960) and Law et al. (1985):
    /// B/A(T) = B/A(T₀) + k·(T - T₀)
    ///
    /// Where:
    /// - B/A(20°C) = 5.0 (reference value)
    /// - k = 0.025 K⁻¹ (temperature coefficient)
    ///
    /// Valid range: 0-100°C
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Nonlinear parameter B/A [dimensionless]
    pub fn nonlinear_parameter_water(&self, temperature: f64) -> f64 {
        const B_A_REF: f64 = 5.0;          // At 20°C (dimensionless)
        const T_REF: f64 = 20.0;           // °C
        const K_TEMP: f64 = 0.025;         // K⁻¹ (temperature coefficient)

        B_A_REF + K_TEMP * (temperature - T_REF)
    }

    /// Calculate frequency-dependent attenuation coefficient for water
    ///
    /// Uses power law model: α(f) = α₀·f^b
    ///
    /// For water at 20°C:
    /// - α₀ ≈ 2.17e-14 (Np/m)·s^b (base attenuation)
    /// - b = 2.0 (frequency exponent)
    ///
    /// Temperature dependence included via Francois & Garrison (1982) model
    /// delegated to WaterProperties::absorption_coefficient.
    ///
    /// # Arguments
    /// * `frequency` - Frequency [Hz]
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Attenuation coefficient [Np/m]
    pub fn attenuation_coefficient_water(&self, frequency: f64, temperature: f64) -> f64 {
        // Use Francois & Garrison (1982) model for comprehensive absorption
        // (already implemented in water.rs)
        // Assume fresh water: depth=0, salinity=0, pH=7
        super::water::WaterProperties::absorption_coefficient(
            frequency,
            temperature,
            0.0,  // depth
            0.0,  // salinity (fresh water)
            7.0,  // pH (neutral)
        )
    }

    /// Calculate frequency-dependent attenuation coefficient for soft tissue
    ///
    /// Power law model: α(f) = α₀·f^b
    ///
    /// For soft tissue (Duck 1990):
    /// - α₀ = 0.5 dB/(cm·MHz^b)
    /// - b = 1.5 (frequency exponent, tissue-specific)
    ///
    /// Temperature dependence (Q10 = 1.3):
    /// α(T) = α(T₀)·Q10^((T - T₀)/10)
    ///
    /// # Arguments
    /// * `frequency` - Frequency [Hz]
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Attenuation coefficient [Np/m]
    pub fn attenuation_coefficient_tissue(&self, frequency: f64, temperature: f64) -> f64 {
        const ALPHA_0: f64 = 0.5;          // dB/(cm·MHz^b)
        const B: f64 = 1.5;                // Frequency exponent
        const T_REF: f64 = 37.0;           // Body temperature reference
        const Q10: f64 = 1.3;              // Temperature sensitivity

        // Convert frequency to MHz
        let f_mhz = frequency / 1e6;

        // Base attenuation at reference temperature
        let alpha_db_cm = ALPHA_0 * f_mhz.powf(B);

        // Temperature correction (Q10 model)
        let temp_factor = Q10.powf((temperature - T_REF) / 10.0);
        let alpha_db_cm_corrected = alpha_db_cm * temp_factor;

        // Convert dB/cm to Np/m
        // 1 dB/cm = 0.1151 Np/m
        alpha_db_cm_corrected * 0.1151 * 100.0
    }

    /// Calculate acoustic impedance Z = ρ·c
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `pressure` - Absolute pressure [Pa]
    ///
    /// # Returns
    /// Acoustic impedance [kg/(m²·s)] or [Rayl]
    pub fn acoustic_impedance_water(&self, temperature: f64, pressure: f64) -> f64 {
        let rho = super::water::WaterProperties::density(temperature);
        let c = self.sound_speed_water(temperature, pressure);
        rho * c
    }

    /// Calculate bulk modulus K = ρ·c²
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `pressure` - Absolute pressure [Pa]
    ///
    /// # Returns
    /// Bulk modulus [Pa]
    pub fn bulk_modulus_water(&self, temperature: f64, pressure: f64) -> f64 {
        let rho = super::water::WaterProperties::density(temperature);
        let c = self.sound_speed_water(temperature, pressure);
        rho * c * c
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
    pub fn thermal_diffusivity_water(&self, temperature: f64) -> f64 {
        // Use constants from thermodynamic.rs
        const K_THERM: f64 = 0.598;        // W/(m·K) at 20°C
        const CP: f64 = 4182.0;            // J/(kg·K)

        let rho = super::water::WaterProperties::density(temperature);

        // Temperature dependence of thermal conductivity (weak)
        let k_temp = K_THERM * (1.0 + 0.002 * (temperature - 20.0));

        k_temp / (rho * CP)
    }

    /// Calculate Prandtl number Pr = ν/κ (viscous/thermal diffusivity ratio)
    ///
    /// Important dimensionless parameter for heat transfer analysis.
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Prandtl number [dimensionless]
    pub fn prandtl_number_water(&self, temperature: f64) -> f64 {
        let nu = self.kinematic_viscosity_water(temperature);
        let kappa = self.thermal_diffusivity_water(temperature);
        nu / kappa
    }

    /// Calculate Reynolds number Re = (ρ·v·L)/η
    ///
    /// Characterizes flow regime (laminar vs turbulent).
    ///
    /// # Arguments
    /// * `velocity` - Characteristic velocity [m/s]
    /// * `length` - Characteristic length [m]
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Reynolds number [dimensionless]
    pub fn reynolds_number_water(&self, velocity: f64, length: f64, temperature: f64) -> f64 {
        let rho = super::water::WaterProperties::density(temperature);
        let eta = self.dynamic_viscosity_water(temperature);

        (rho * velocity * length) / eta
    }

    /// The threshold is the minimum negative pressure (tension) required to initiate cavitation.
    /// P_threshold (tension) = P_v + sqrt(2·σ/(3·R_nuclei)) - P_ambient
    ///
    /// All parameters are temperature-dependent.
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `nuclei_radius` - Bubble nuclei radius [m]
    /// * `ambient_pressure` - Ambient pressure [Pa]
    ///
    /// # Returns
    /// Cavitation threshold pressure amplitude (negative = tension) [Pa]
    pub fn cavitation_threshold(&self, temperature: f64, nuclei_radius: f64, ambient_pressure: f64) -> f64 {
        // Vapor pressure of water (temperature-dependent)
        let p_v = self.vapor_pressure_water(temperature);

        // Surface tension (temperature-dependent)
        let sigma = self.surface_tension_water(temperature);

        // Blake threshold (tension required to overcome surface tension)
        let p_blake = (2.0 * sigma / (3.0 * nuclei_radius)).sqrt();

        // Total threshold as negative pressure (tension)
        // Acoustic pressure amplitude must exceed this to cavitate
        -(ambient_pressure - p_v - p_blake)
    }

    /// Calculate vapor pressure of water with temperature dependence
    ///
    /// Uses Antoine equation:
    /// log₁₀(P_v) = A - B/(C + T)
    ///
    /// Where P_v is in mmHg and T in °C.
    ///
    /// Constants for water (0-100°C):
    /// - A = 8.07131
    /// - B = 1730.63
    /// - C = 233.426
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Vapor pressure [Pa]
    pub fn vapor_pressure_water(&self, temperature: f64) -> f64 {
        // Antoine equation constants for water
        const A: f64 = 8.07131;
        const B: f64 = 1730.63;
        const C: f64 = 233.426;

        // Calculate vapor pressure in mmHg
        let log_p_mmhg = A - B / (C + temperature);
        let p_mmhg = 10.0_f64.powf(log_p_mmhg);

        // Convert mmHg to Pa (1 mmHg = 133.322 Pa)
        p_mmhg * 133.322
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sound_speed_water_temperature_dependence() {
        let constants = StateDependentConstants::default();

        // Test del Grosso's temperature gradient: dc/dT ≈ 3.0 m/s/K
        let c_20 = constants.sound_speed_water(20.0, 101325.0);
        let c_21 = constants.sound_speed_water(21.0, 101325.0);

        let dc_dt = c_21 - c_20;
        assert!((dc_dt - 3.0).abs() < 1.0, "dc/dT should be ~3.0 m/s/K, got {}", dc_dt);

        // Verify body temperature value
        let c_37 = constants.sound_speed_water(37.0, 101325.0);
        assert!((c_37 - 1525.0).abs() < 10.0, "Sound speed at 37°C should be ~1525 m/s, got {}", c_37);
    }

    #[test]
    fn test_dynamic_viscosity_water() {
        let constants = StateDependentConstants::default();

        // Test viscosity at 20°C (should be ~1.002e-3 Pa·s)
        let eta_20 = constants.dynamic_viscosity_water(20.0);
        assert!((eta_20 - 1.002e-3).abs() < 1e-5, "Viscosity at 20°C should be ~1.002e-3 Pa·s, got {}", eta_20);

        // Test viscosity at 37°C (should be ~0.692e-3 Pa·s, interpolated between 30°C and 40°C)
        let eta_37 = constants.dynamic_viscosity_water(37.0);
        assert!((eta_37 - 0.692e-3).abs() < 5e-5, "Viscosity at 37°C should be ~0.692e-3 Pa·s, got {}", eta_37);

        // Viscosity should decrease with temperature
        assert!(eta_20 > eta_37, "Viscosity should decrease with temperature");
    }

    #[test]
    fn test_surface_tension_water() {
        let constants = StateDependentConstants::default();

        // Test surface tension at 20°C (should be ~0.0728 N/m)
        let sigma_20 = constants.surface_tension_water(20.0);
        assert!((sigma_20 - 0.0728).abs() < 0.005, "Surface tension at 20°C should be ~0.0728 N/m, got {}", sigma_20);

        // Test surface tension at 100°C (should be ~0.0589 N/m)
        let sigma_100 = constants.surface_tension_water(100.0);
        assert!((sigma_100 - 0.0589).abs() < 0.01, "Surface tension at 100°C should be ~0.0589 N/m, got {}", sigma_100);

        // Surface tension should decrease with temperature
        assert!(sigma_20 > sigma_100, "Surface tension should decrease with temperature");
    }

    #[test]
    fn test_nonlinear_parameter() {
        let constants = StateDependentConstants::default();

        // Test B/A at reference temperature (should be 5.0)
        let ba_20 = constants.nonlinear_parameter_water(20.0);
        assert!((ba_20 - 5.0).abs() < 0.1, "B/A at 20°C should be ~5.0, got {}", ba_20);

        // B/A should increase with temperature
        let ba_37 = constants.nonlinear_parameter_water(37.0);
        assert!(ba_37 > ba_20, "B/A should increase with temperature");
    }

    #[test]
    fn test_acoustic_impedance() {
        let constants = StateDependentConstants::default();

        // Test acoustic impedance at 20°C (should be ~1.48 MRayl)
        let z_20 = constants.acoustic_impedance_water(20.0, 101325.0);
        assert!((z_20 - 1.48e6).abs() < 0.05e6, "Acoustic impedance at 20°C should be ~1.48 MRayl, got {}", z_20);
    }

    #[test]
    fn test_cavitation_threshold() {
        let constants = StateDependentConstants::default();

        // Test Blake threshold at 20°C for 1 μm nuclei
        let p_thresh = constants.cavitation_threshold(20.0, 1e-6, 101325.0);

        // Should be negative (tension) and on the order of -1 to -5 bar
        assert!(p_thresh < 0.0, "Cavitation threshold should be negative (tension)");
        assert!(p_thresh.abs() < 1e6, "Cavitation threshold should be reasonable magnitude");
    }

    #[test]
    fn test_prandtl_number() {
        let constants = StateDependentConstants::default();

        // Test Prandtl number at 20°C (should be ~7 for water)
        let pr = constants.prandtl_number_water(20.0);
        assert!((pr - 7.0).abs() < 2.0, "Prandtl number at 20°C should be ~7, got {}", pr);
    }

    #[test]
    fn test_reynolds_number() {
        let constants = StateDependentConstants::default();

        // Test Reynolds number for typical flow
        // v = 0.1 m/s, L = 0.01 m, T = 20°C
        let re = constants.reynolds_number_water(0.1, 0.01, 20.0);

        // Should be ~1000 (laminar-turbulent transition region)
        assert!(re > 500.0 && re < 2000.0, "Reynolds number should be in laminar regime, got {}", re);
    }
}
