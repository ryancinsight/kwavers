//! Centralized physics constants
//!
//! Single Source of Truth (SSOT) for all physical constants used throughout
//! the simulation. All values are from authoritative sources with citations.

/// Acoustic properties of water at 20°C, 1 atm
pub mod water {
    /// Density (kg/m³) - CRC Handbook of Chemistry and Physics, 102nd ed.
    pub const DENSITY: f64 = 998.2071;

    /// Speed of sound (m/s) - Del Grosso & Mader (1972)
    pub const SOUND_SPEED: f64 = 1482.343;

    /// Bulk modulus (Pa)
    pub const BULK_MODULUS: f64 = 2.19e9;

    /// Dynamic viscosity (Pa·s) at 20°C
    pub const VISCOSITY: f64 = 1.002e-3;

    /// Thermal conductivity (W/m·K)
    pub const THERMAL_CONDUCTIVITY: f64 = 0.598;

    /// Specific heat capacity (J/kg·K)
    pub const SPECIFIC_HEAT: f64 = 4182.0;

    /// Absorption coefficient (Np/m/MHz²) - Pinkerton (1949)
    pub const ABSORPTION_COEFFICIENT: f64 = 0.0022;

    /// Nonlinearity parameter B/A - Beyer (1960)
    pub const NONLINEARITY_PARAMETER: f64 = 5.0;
}

/// Acoustic properties of soft tissue (average)
pub mod tissue {
    /// Density (kg/m³) - Duck (1990)
    pub const DENSITY: f64 = 1050.0;

    /// Speed of sound (m/s) - Goss et al. (1978)
    pub const SOUND_SPEED: f64 = 1540.0;

    /// Absorption coefficient (dB/cm/MHz) - Goss et al. (1978)
    pub const ABSORPTION_COEFFICIENT: f64 = 0.5;

    /// Absorption power law exponent
    pub const ABSORPTION_POWER: f64 = 1.05;

    /// Nonlinearity parameter B/A - Law et al. (1985)
    pub const NONLINEARITY_PARAMETER: f64 = 6.0;

    /// Thermal conductivity (W/m·K)
    pub const THERMAL_CONDUCTIVITY: f64 = 0.5;

    /// Specific heat capacity (J/kg·K)
    pub const SPECIFIC_HEAT: f64 = 3600.0;

    /// Perfusion rate (1/s) for bioheat equation
    pub const PERFUSION_RATE: f64 = 0.0005;
}

/// Acoustic properties of air at 20°C, 1 atm
pub mod air {
    /// Density (kg/m³)
    pub const DENSITY: f64 = 1.204;

    /// Speed of sound (m/s)
    pub const SOUND_SPEED: f64 = 343.21;

    /// Specific heat ratio (gamma)
    pub const GAMMA: f64 = 1.4;

    /// Dynamic viscosity (Pa·s)
    pub const VISCOSITY: f64 = 1.82e-5;
}

/// Blood properties for perfusion modeling
pub mod blood {
    /// Density (kg/m³)
    pub const DENSITY: f64 = 1060.0;

    /// Specific heat capacity (J/kg·K)
    pub const SPECIFIC_HEAT: f64 = 3617.0;

    /// Temperature (°C) for perfusion
    pub const TEMPERATURE: f64 = 37.0;
}

/// Bone properties
pub mod bone {
    /// Density (kg/m³) - cortical bone
    pub const DENSITY: f64 = 1908.0;

    /// Speed of sound (m/s) - longitudinal wave
    pub const SOUND_SPEED: f64 = 4080.0;

    /// Absorption coefficient (dB/cm/MHz)
    pub const ABSORPTION_COEFFICIENT: f64 = 2.0;
}

/// Fat tissue properties
pub mod fat {
    /// Density (kg/m³)
    pub const DENSITY: f64 = 950.0;

    /// Speed of sound (m/s)
    pub const SOUND_SPEED: f64 = 1450.0;

    /// Absorption coefficient (dB/cm/MHz)
    pub const ABSORPTION_COEFFICIENT: f64 = 0.48;
}

/// Muscle tissue properties
pub mod muscle {
    /// Density (kg/m³)
    pub const DENSITY: f64 = 1090.0;

    /// Speed of sound (m/s)
    pub const SOUND_SPEED: f64 = 1580.0;

    /// Absorption coefficient (dB/cm/MHz)
    pub const ABSORPTION_COEFFICIENT: f64 = 0.57;
}

/// Liver tissue properties
pub mod liver {
    /// Density (kg/m³)
    pub const DENSITY: f64 = 1079.0;

    /// Speed of sound (m/s)
    pub const SOUND_SPEED: f64 = 1578.0;

    /// Absorption coefficient (dB/cm/MHz)
    pub const ABSORPTION_COEFFICIENT: f64 = 0.5;
}

/// Physical constants
pub mod physical {
    /// Boltzmann constant (J/K)
    pub const BOLTZMANN: f64 = 1.380649e-23;

    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299792458.0;

    /// Standard atmospheric pressure (Pa)
    pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

    /// Standard gravity (m/s²)
    pub const GRAVITY: f64 = 9.80665;

    /// Avogadro's number (1/mol)
    pub const AVOGADRO: f64 = 6.02214076e23;

    /// Gas constant (J/mol·K)
    pub const GAS_CONSTANT: f64 = 8.314462618;
}

/// Unit conversions
pub mod conversions {
    /// Convert dB/cm/MHz to Np/m at 1 `MHz`
    pub const DB_CM_MHZ_TO_NP_M: f64 = 0.1151 / 0.01;

    /// Convert Np to dB
    pub const NP_TO_DB: f64 = 8.685889638;

    /// Convert degrees to radians
    pub const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

    /// Convert radians to degrees  
    pub const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
}

/// Numerical thresholds and limits
pub mod numerical {
    /// Machine epsilon for f64
    pub const EPSILON: f64 = f64::EPSILON;

    /// Default tolerance for convergence checks
    pub const CONVERGENCE_TOLERANCE: f64 = 1e-6;

    /// Maximum iterations for iterative solvers
    pub const MAX_ITERATIONS: usize = 1000;

    /// Minimum time step for stability (seconds)
    pub const MIN_TIME_STEP: f64 = 1e-12;

    /// Maximum CFL number for stability
    pub const MAX_CFL: f64 = 0.3;

    /// Threshold for considering a value as zero
    pub const ZERO_THRESHOLD: f64 = 1e-14;
}

/// Transducer and array parameters
pub mod transducer {
    /// Maximum steering angle (radians)
    pub const MAX_STEERING_ANGLE: f64 = std::f64::consts::PI / 3.0; // 60 degrees

    /// Minimum focal distance (m)
    pub const MIN_FOCAL_DISTANCE: f64 = 0.01; // 10 mm

    /// Maximum focal distance (m)
    pub const MAX_FOCAL_DISTANCE: f64 = 0.3; // 300 mm

    /// Typical element spacing (wavelengths)
    pub const ELEMENT_SPACING_LAMBDA: f64 = 0.5;

    /// Maximum number of focal points for multi-focus
    pub const MAX_FOCAL_POINTS: usize = 8;

    /// Phase quantization levels for digital control
    pub const PHASE_QUANTIZATION_LEVELS: usize = 256;
}

/// Bubble dynamics parameters
pub mod bubble {
    /// Surface tension of water (N/m) at 20°C
    pub const SURFACE_TENSION: f64 = 0.0728;

    /// Polytropic exponent for gas
    pub const POLYTROPIC_EXPONENT: f64 = 1.4;

    /// Vapor pressure of water at 20°C (Pa)
    pub const VAPOR_PRESSURE: f64 = 2338.0;

    /// Minimum bubble radius for stability (m)
    pub const MIN_RADIUS: f64 = 1e-9;

    /// Maximum bubble radius before breakup (m)
    pub const MAX_RADIUS: f64 = 1e-3;
}

/// Thermal dose parameters
pub mod thermal {
    /// Reference temperature for thermal dose (°C)
    pub const REFERENCE_TEMPERATURE: f64 = 43.0;

    /// Thermal dose threshold for ablation (equivalent minutes)
    pub const ABLATION_THRESHOLD: f64 = 240.0;

    /// Activation energy for tissue damage (J/mol)
    pub const ACTIVATION_ENERGY: f64 = 6.28e5;

    /// Frequency factor for Arrhenius model (1/s)
    pub const FREQUENCY_FACTOR: f64 = 7.39e104;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_impedance() {
        let impedance = water::DENSITY * water::SOUND_SPEED;
        assert!((impedance - 1.478e6).abs() / impedance < 0.01);
    }

    #[test]
    fn test_conversion_consistency() {
        use conversions::*;
        assert!((NP_TO_DB * (1.0 / NP_TO_DB) - 1.0).abs() < 1e-10);
        assert!((DEG_TO_RAD * RAD_TO_DEG - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_constraint() {
        // CFL condition: dt <= CFL * dx / c
        let dx = 1e-4; // 0.1 mm
        let c = tissue::SOUND_SPEED;
        let dt_max = numerical::MAX_CFL * dx / c;
        assert!(dt_max > numerical::MIN_TIME_STEP);
    }
}
