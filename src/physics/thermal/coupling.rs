//! Thermal-acoustic coupling effects
//!
//! Implements bidirectional coupling between acoustic waves and thermal fields:
//! - Acoustic → Thermal: Viscous absorption generates heat
//! - Thermal → Acoustic: Temperature affects acoustic properties
//!
//! References:
//! - Hamilton & Blackstock (1998) "Nonlinear Acoustics"
//! - Nyborg (1988) "Acoustic streaming in ultrasonic therapy"
//! - ter Haar & Coussios (2007) "High intensity focused ultrasound"
//! - Zeqiri (2008) "Cavitation and ultrasonic surgery"

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Acoustic heating source
///
/// Represents heat generation from ultrasound absorption (I²-type heating)
#[derive(Debug, Clone, Copy)]
pub struct AcousticHeatingSource {
    /// Absorption coefficient [Np/m]
    pub absorption_coefficient: f64,
    /// Acoustic intensity [W/m²]
    pub intensity: f64,
}

impl AcousticHeatingSource {
    /// Create acoustic heating source
    pub fn new(absorption_coefficient: f64, intensity: f64) -> Self {
        Self {
            absorption_coefficient,
            intensity,
        }
    }

    /// Heat source power [W/m³]
    ///
    /// Q = 2·α·I
    /// where:
    /// - α = absorption coefficient [Np/m]
    /// - I = acoustic intensity [W/m²]
    ///
    /// This comes from the acoustic power balance equation.
    pub fn power(&self) -> f64 {
        2.0 * self.absorption_coefficient * self.intensity
    }

    /// Heat source power at depth z
    ///
    /// Q(z) = 2·α·I·exp(-2αz)
    /// Accounts for attenuation of acoustic intensity with depth.
    pub fn power_at_depth(&self, depth: f64) -> f64 {
        2.0 * self.absorption_coefficient
            * self.intensity
            * (-2.0 * self.absorption_coefficient * depth).exp()
    }
}

/// Temperature dependence of acoustic properties
///
/// Most tissue properties change with temperature:
/// - Sound speed: ∂c/∂T ≈ 2 m/s/°C
/// - Density: ∂ρ/∂T ≈ -0.5 kg/m³/°C
/// - Absorption: ∂α/∂T varies by tissue
#[derive(Debug, Clone, Copy)]
pub struct TemperatureCoefficients {
    /// Sound speed temperature coefficient [m/s/°C]
    pub sound_speed_coeff: f64,
    /// Density temperature coefficient [kg/m³/°C]
    pub density_coeff: f64,
    /// Absorption temperature coefficient [Np/m/°C]
    pub absorption_coeff: f64,
}

impl TemperatureCoefficients {
    /// Create custom temperature coefficients
    pub fn new(sound_speed_coeff: f64, density_coeff: f64, absorption_coeff: f64) -> Self {
        Self {
            sound_speed_coeff,
            density_coeff,
            absorption_coeff,
        }
    }

    /// Generic soft tissue coefficients
    /// Reference: Duck (1990), Szabo (2004)
    pub fn soft_tissue() -> Self {
        Self {
            sound_speed_coeff: 2.0,  // [m/s/°C]
            density_coeff: -0.5,     // [kg/m³/°C]
            absorption_coeff: 0.015, // [Np/m/°C]
        }
    }

    /// Water coefficients
    /// Reference: IEC 61161:2013
    pub fn water() -> Self {
        Self {
            sound_speed_coeff: 4.0, // [m/s/°C]
            density_coeff: -0.2,    // [kg/m³/°C]
            absorption_coeff: 0.0,  // [Np/m/°C] (negligible)
        }
    }

    /// Blood coefficients
    /// Reference: Gordon et al. (2009)
    pub fn blood() -> Self {
        Self {
            sound_speed_coeff: 2.5, // [m/s/°C]
            density_coeff: -0.6,    // [kg/m³/°C]
            absorption_coeff: 0.02, // [Np/m/°C]
        }
    }

    /// Bone coefficients
    /// Reference: Duck (1990)
    pub fn bone() -> Self {
        Self {
            sound_speed_coeff: 1.0,  // [m/s/°C]
            density_coeff: -0.1,     // [kg/m³/°C]
            absorption_coeff: 0.005, // [Np/m/°C]
        }
    }

    /// Sound speed at temperature
    pub fn sound_speed(&self, base_sound_speed: f64, temperature: f64, reference_temp: f64) -> f64 {
        base_sound_speed + self.sound_speed_coeff * (temperature - reference_temp)
    }

    /// Density at temperature
    pub fn density(&self, base_density: f64, temperature: f64, reference_temp: f64) -> f64 {
        base_density + self.density_coeff * (temperature - reference_temp)
    }

    /// Absorption at temperature
    pub fn absorption(&self, base_absorption: f64, temperature: f64, reference_temp: f64) -> f64 {
        (base_absorption + self.absorption_coeff * (temperature - reference_temp)).max(0.0)
    }
}

impl Default for TemperatureCoefficients {
    fn default() -> Self {
        Self::soft_tissue()
    }
}

/// Acoustic streaming effects
///
/// Acoustic streaming is the steady flow of fluid induced by acoustic waves.
/// It contributes to heat mixing and can enhance thermal effects.
///
/// Reference: Nyborg (1988) "Acoustic streaming in ultrasonic therapy"
#[derive(Debug, Clone, Copy)]
pub struct AcousticStreaming {
    /// Acoustic intensity [W/m²]
    pub intensity: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Fluid density [kg/m³]
    pub density: f64,
}

impl AcousticStreaming {
    /// Create streaming effect
    pub fn new(intensity: f64, sound_speed: f64, density: f64) -> Self {
        Self {
            intensity,
            sound_speed,
            density,
        }
    }

    /// Streaming velocity [m/s]
    ///
    /// v_stream ~ I / (ρ·c)²
    /// Derived from radiation stress tensor
    pub fn velocity(&self) -> f64 {
        self.intensity / ((self.density * self.sound_speed).powi(2))
    }

    /// Streaming power (energy flux) [W/m³]
    pub fn power(&self) -> f64 {
        self.intensity.powi(2) / (self.density * self.sound_speed)
    }

    /// Enhance thermal mixing (effective diffusivity increase)
    ///
    /// Acoustic streaming can enhance thermal transport.
    /// Effective diffusivity increase: Δα ~ (L·v_stream)²/t_acoustic
    /// where L is characteristic length scale
    pub fn enhanced_diffusivity(&self, characteristic_length: f64) -> f64 {
        let v = self.velocity();
        let period = 1.0 / 1e6; // Assume 1 MHz frequency
        (characteristic_length * v).powi(2) / period
    }
}

/// Nonlinear acoustic effects (Second-harmonic generation)
///
/// Nonlinear acoustics contribute to heating through shock formation
/// and generation of higher harmonics that are more readily absorbed.
#[derive(Debug, Clone, Copy)]
pub struct NonlinearHeating {
    /// Nonlinearity parameter (B/A)
    pub nonlinearity_parameter: f64,
    /// Acoustic pressure [Pa]
    pub pressure: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Density [kg/m³]
    pub density: f64,
}

impl NonlinearHeating {
    /// Create nonlinear heating source
    pub fn new(nonlinearity_parameter: f64, pressure: f64, sound_speed: f64, density: f64) -> Self {
        Self {
            nonlinearity_parameter,
            pressure,
            sound_speed,
            density,
        }
    }

    /// Additional heating from nonlinearity [W/m³]
    ///
    /// P_nl ~ (B/A)·P²·f² / (ρ·c³)
    /// where f is frequency (implicitly in pressure amplitude)
    pub fn power(&self) -> f64 {
        let c3 = self.sound_speed.powi(3);
        self.nonlinearity_parameter * self.pressure.powi(2) / (self.density * c3)
    }

    /// Shock formation parameter (Mach number for acoustic waves)
    ///
    /// σ = (B/A)·P / (2·ρ·c²)
    /// Indicates propensity for shock formation
    pub fn shock_parameter(&self) -> f64 {
        self.nonlinearity_parameter * self.pressure
            / (2.0 * self.density * self.sound_speed.powi(2))
    }

    /// Is nonlinear regime significant?
    /// (σ > 0.01 generally indicates significant nonlinear effects)
    pub fn is_nonlinear_significant(&self) -> bool {
        self.shock_parameter() > 0.01
    }
}

/// Thermal-acoustic coupling field solver
#[derive(Debug)]
pub struct ThermalAcousticCoupling {
    /// Acoustic heating source
    source: AcousticHeatingSource,
    /// Temperature coefficients
    coefficients: TemperatureCoefficients,
    /// Accumulated acoustic heating [J/m³]
    acoustic_heat: Array3<f64>,
}

impl ThermalAcousticCoupling {
    /// Create coupling solver
    pub fn new(
        absorption_coefficient: f64,
        intensity: f64,
        coefficients: TemperatureCoefficients,
    ) -> Self {
        Self {
            source: AcousticHeatingSource::new(absorption_coefficient, intensity),
            coefficients,
            acoustic_heat: Array3::zeros((1, 1, 1)),
        }
    }

    /// Initialize heat field
    pub fn initialize(&mut self, shape: (usize, usize, usize)) {
        self.acoustic_heat = Array3::zeros(shape);
    }

    /// Update coupling with acoustic and thermal fields
    pub fn update(
        &mut self,
        temperature: &Array3<f64>,
        acoustic_intensity: &Array3<f64>,
        reference_temperature: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = (
            self.acoustic_heat.dim().0,
            self.acoustic_heat.dim().1,
            self.acoustic_heat.dim().2,
        );

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let t = temperature[[i, j, k]];
                    let i_ac = acoustic_intensity[[i, j, k]];

                    // Heat generation from acoustic absorption
                    // Q = 2·α·I where α depends on temperature
                    let alpha = self.coefficients.absorption(
                        self.source.absorption_coefficient,
                        t,
                        reference_temperature,
                    );
                    let heat_rate = 2.0 * alpha * i_ac;

                    // Accumulate heat
                    self.acoustic_heat[[i, j, k]] += heat_rate * dt;
                }
            }
        }

        Ok(())
    }

    /// Get acoustic heating field [W/m³]
    pub fn acoustic_heat(&self) -> &Array3<f64> {
        &self.acoustic_heat
    }

    /// Total acoustic energy deposited [J]
    pub fn total_energy(&self) -> f64 {
        self.acoustic_heat.iter().sum()
    }

    /// Reset coupling field
    pub fn reset(&mut self) {
        self.acoustic_heat.fill(0.0);
    }

    /// Get modified sound speed accounting for temperature
    pub fn sound_speed_at_temperature(
        &self,
        base_sound_speed: f64,
        temperature: f64,
        reference_temperature: f64,
    ) -> f64 {
        self.coefficients
            .sound_speed(base_sound_speed, temperature, reference_temperature)
    }

    /// Get modified density accounting for temperature
    pub fn density_at_temperature(
        &self,
        base_density: f64,
        temperature: f64,
        reference_temperature: f64,
    ) -> f64 {
        self.coefficients
            .density(base_density, temperature, reference_temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acoustic_heating_source() {
        let source = AcousticHeatingSource::new(0.5, 1e4); // 500 Np/m, 10 kW/m²
        let power = source.power();
        assert!(power > 0.0);
    }

    #[test]
    fn test_heating_depth_attenuation() {
        let source = AcousticHeatingSource::new(0.5, 1e4);
        let power_0 = source.power_at_depth(0.0);
        let power_1cm = source.power_at_depth(0.01);

        // Power should decrease with depth
        assert!(power_1cm < power_0);
    }

    #[test]
    fn test_temperature_coefficients_soft_tissue() {
        let coeff = TemperatureCoefficients::soft_tissue();

        // Base properties at 37°C
        let c0 = 1540.0;
        let rho0 = 1050.0;
        let alpha0 = 0.5;

        // At 40°C (3°C higher)
        let c_40 = coeff.sound_speed(c0, 40.0, 37.0);
        let rho_40 = coeff.density(rho0, 40.0, 37.0);
        let alpha_40 = coeff.absorption(alpha0, 40.0, 37.0);

        // Sound speed increases
        assert!(c_40 > c0);
        // Density decreases
        assert!(rho_40 < rho0);
        // Absorption increases
        assert!(alpha_40 > alpha0);
    }

    #[test]
    fn test_acoustic_streaming_velocity() {
        let streaming = AcousticStreaming::new(1e3, 1500.0, 1050.0); // 1 kW/m²
        let v = streaming.velocity();
        assert!(v > 0.0);
    }

    #[test]
    fn test_nonlinear_heating() {
        let nl = NonlinearHeating::new(
            5.0, // B/A = 5
            1e5, // 100 kPa
            1500.0, 1050.0,
        );
        let power = nl.power();
        assert!(power > 0.0);

        let shock = nl.shock_parameter();
        assert!(shock > 0.0);
    }

    #[test]
    fn test_nonlinear_regime_detection() {
        // Linear regime
        let nl_linear = NonlinearHeating::new(5.0, 1e4, 1500.0, 1050.0);
        assert!(!nl_linear.is_nonlinear_significant());

        // Nonlinear regime
        let nl_nonlinear = NonlinearHeating::new(5.0, 5e5, 1500.0, 1050.0);
        assert!(nl_nonlinear.is_nonlinear_significant());
    }

    #[test]
    fn test_thermal_acoustic_coupling() {
        let mut coupling =
            ThermalAcousticCoupling::new(0.5, 1e4, TemperatureCoefficients::soft_tissue());
        coupling.initialize((5, 5, 5));

        let temperature = Array3::from_elem((5, 5, 5), 37.0);
        let intensity = Array3::from_elem((5, 5, 5), 1e4);

        let result = coupling.update(&temperature, &intensity, 37.0, 0.1);
        assert!(result.is_ok());

        let energy = coupling.total_energy();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_coupling_temperature_effects_on_properties() {
        let coupling =
            ThermalAcousticCoupling::new(0.5, 1e4, TemperatureCoefficients::soft_tissue());

        let c0 = 1540.0;
        let c_hot = coupling.sound_speed_at_temperature(c0, 45.0, 37.0);
        let rho0 = 1050.0;
        let rho_hot = coupling.density_at_temperature(rho0, 45.0, 37.0);

        // Temperature increases both sound speed and decreases density
        assert!(c_hot > c0);
        assert!(rho_hot < rho0);
    }

    #[test]
    fn test_temperature_coefficient_variants() {
        let soft = TemperatureCoefficients::soft_tissue();
        let water = TemperatureCoefficients::water();
        let blood = TemperatureCoefficients::blood();
        let bone = TemperatureCoefficients::bone();

        // Each should have different coefficients
        assert_ne!(soft.sound_speed_coeff, water.sound_speed_coeff);
        assert_ne!(blood.absorption_coeff, bone.absorption_coeff);
    }

    #[test]
    fn test_acoustic_heating_zero_absorption() {
        let source = AcousticHeatingSource::new(0.0, 1e5);
        assert_eq!(source.power(), 0.0);
    }

    #[test]
    fn test_acoustic_heating_zero_intensity() {
        let source = AcousticHeatingSource::new(0.5, 0.0);
        assert_eq!(source.power(), 0.0);
    }
}
