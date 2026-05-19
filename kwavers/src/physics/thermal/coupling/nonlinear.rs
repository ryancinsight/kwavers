//! Nonlinear acoustic effects (Second-harmonic generation)
//!
//! Nonlinear acoustics contribute to heating through shock formation
//! and generation of higher harmonics that are more readily absorbed.

/// Nonlinear acoustic effects generating secondary absorption
#[derive(Debug, Clone, Copy)]
pub struct NonlinearHeating {
    /// Nonlinearity parameter (B/A)
    pub nonlinearity_parameter: f64,
    /// Acoustic pressure amplitude (Pa)
    pub pressure: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Driving frequency (Hz)
    pub frequency: f64,
}

impl NonlinearHeating {
    /// Create nonlinear heating source
    #[must_use]
    pub fn new(
        nonlinearity_parameter: f64,
        pressure: f64,
        sound_speed: f64,
        density: f64,
        frequency: f64,
    ) -> Self {
        Self {
            nonlinearity_parameter,
            pressure,
            sound_speed,
            density,
            frequency,
        }
    }

    /// Additional heating from nonlinearity [W/m³]
    ///
    /// Q_nl = (B/A)·P²·ω² / (ρ·c³)
    ///
    /// Derived from the second-order Westervelt source term: the generated
    /// harmonics are absorbed proportional to ω² (Hamilton & Blackstock 1998,
    /// §4.3; Sehgal & Greenleaf 1984).
    #[must_use]
    pub fn power(&self) -> f64 {
        use std::f64::consts::PI;
        let omega = 2.0 * PI * self.frequency;
        let c3 = self.sound_speed.powi(3);
        self.nonlinearity_parameter * self.pressure.powi(2) * omega.powi(2)
            / (self.density * c3)
    }

    /// Shock formation parameter (Mach number for acoustic waves)
    ///
    /// σ = (B/A)·P / (2·ρ·c²)
    /// Indicates propensity for shock formation
    #[must_use]
    pub fn shock_parameter(&self) -> f64 {
        self.nonlinearity_parameter * self.pressure
            / (2.0 * self.density * self.sound_speed.powi(2))
    }

    /// Is nonlinear regime significant?
    /// (σ > 1e-4 generally indicates significant nonlinear effects)
    /// This threshold is based on the acoustic nonlinearity coefficient (B/A)
    /// being about 5-8 for most tissues, combined with typical therapeutic pressures
    #[must_use]
    pub fn is_nonlinear_significant(&self) -> bool {
        self.shock_parameter() > 1e-4
    }
}
