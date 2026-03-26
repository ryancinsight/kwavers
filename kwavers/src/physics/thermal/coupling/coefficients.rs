//! Temperature dependence of acoustic properties
//!
//! Most tissue properties change with temperature:
//! - Sound speed: ∂c/∂T ≈ 2 m/s/°C
//! - Density: ∂ρ/∂T ≈ -0.5 kg/m³/°C
//! - Absorption: ∂α/∂T varies by tissue

/// Temperature coefficients for acoustic properties
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
    #[must_use]
    pub fn new(sound_speed_coeff: f64, density_coeff: f64, absorption_coeff: f64) -> Self {
        Self {
            sound_speed_coeff,
            density_coeff,
            absorption_coeff,
        }
    }

    /// Generic soft tissue coefficients
    /// Reference: Duck (1990), Szabo (2004)
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            sound_speed_coeff: 2.0,  // [m/s/°C]
            density_coeff: -0.5,     // [kg/m³/°C]
            absorption_coeff: 0.015, // [Np/m/°C]
        }
    }

    /// Water coefficients
    /// Reference: IEC 61161:2013
    #[must_use]
    pub fn water() -> Self {
        Self {
            sound_speed_coeff: 4.0, // [m/s/°C]
            density_coeff: -0.2,    // [kg/m³/°C]
            absorption_coeff: 0.0,  // [Np/m/°C] (negligible)
        }
    }

    /// Blood coefficients
    /// Reference: Gordon et al. (2009)
    #[must_use]
    pub fn blood() -> Self {
        Self {
            sound_speed_coeff: 2.5, // [m/s/°C]
            density_coeff: -0.6,    // [kg/m³/°C]
            absorption_coeff: 0.02, // [Np/m/°C]
        }
    }

    /// Bone coefficients
    /// Reference: Duck (1990)
    #[must_use]
    pub fn bone() -> Self {
        Self {
            sound_speed_coeff: 1.0,  // [m/s/°C]
            density_coeff: -0.1,     // [kg/m³/°C]
            absorption_coeff: 0.005, // [Np/m/°C]
        }
    }

    /// Sound speed at temperature
    #[must_use]
    pub fn sound_speed(&self, base_sound_speed: f64, temperature: f64, reference_temp: f64) -> f64 {
        base_sound_speed + self.sound_speed_coeff * (temperature - reference_temp)
    }

    /// Density at temperature
    #[must_use]
    pub fn density(&self, base_density: f64, temperature: f64, reference_temp: f64) -> f64 {
        base_density + self.density_coeff * (temperature - reference_temp)
    }

    /// Absorption at temperature
    #[must_use]
    pub fn absorption(&self, base_absorption: f64, temperature: f64, reference_temp: f64) -> f64 {
        (base_absorption + self.absorption_coeff * (temperature - reference_temp)).max(0.0)
    }
}

impl Default for TemperatureCoefficients {
    fn default() -> Self {
        Self::soft_tissue()
    }
}
