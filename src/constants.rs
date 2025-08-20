//! Physical and numerical constants used throughout the codebase

/// Physics constants submodule
pub mod physics {
    // Water properties at standard conditions (20°C, 1 atm)
    pub const WATER_DENSITY: f64 = 1000.0;  // kg/m³
    pub const WATER_SOUND_SPEED: f64 = 1500.0;  // m/s
    pub const WATER_ATTENUATION: f64 = 0.0022;  // Np/m/MHz
    pub const WATER_NONLINEARITY: f64 = 3.5;  // B/A parameter
    pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.6;  // W/(m·K)
    pub const WATER_SPECIFIC_HEAT: f64 = 4180.0;  // J/(kg·K)
    pub const WATER_REFRACTIVE_INDEX: f64 = 1.33;
    pub const WATER_GRUNEISEN: f64 = 0.12;
    
    // Aliases for compatibility
    pub const DENSITY_WATER: f64 = WATER_DENSITY;
    pub const SOUND_SPEED_WATER: f64 = WATER_SOUND_SPEED;
    
    // Tissue properties (soft tissue average)
    pub const DENSITY_TISSUE: f64 = 1050.0;  // kg/m³
    pub const SOUND_SPEED_TISSUE: f64 = 1540.0;  // m/s
    pub const TISSUE_ATTENUATION: f64 = 0.5;  // dB/cm/MHz
    pub const TISSUE_NONLINEARITY: f64 = 6.0;  // B/A parameter
    
    // Ultrasound parameters
    pub const DEFAULT_ULTRASOUND_FREQUENCY: f64 = 1e6;  // 1 MHz
    pub const STANDARD_PRESSURE_AMPLITUDE: f64 = 1e6;  // 1 MPa
    pub const STANDARD_BEAM_WIDTH: f64 = 0.01;  // 10 mm
    
    // Default mode conversion efficiency
    pub const DEFAULT_MODE_CONVERSION_EFFICIENCY: f64 = 0.3;
    
    // Power law absorption
    pub const DEFAULT_POWER_LAW_EXPONENT: f64 = 1.05;  // Typical for biological tissues
}

// Numerical tolerances
pub const FLOAT_EQUALITY_TOLERANCE: f64 = 1e-10;
pub const NUMERICAL_EPSILON: f64 = 1e-6;
pub const SYMMETRY_TOLERANCE: f64 = 1e-10;
pub const SINC_ARGUMENT_THRESHOLD: f64 = 1e-10;

// CFL and stability
pub const DEFAULT_CFL_SAFETY_FACTOR: f64 = 0.3;
pub const MAX_CFL_NUMBER: f64 = 1.0;

// Water properties at standard conditions (20°C, 1 atm)
pub const WATER_DENSITY: f64 = 1000.0;  // kg/m³
pub const WATER_SOUND_SPEED: f64 = 1500.0;  // m/s
pub const WATER_ATTENUATION: f64 = 0.0022;  // Np/m/MHz
pub const WATER_NONLINEARITY: f64 = 3.5;  // B/A parameter
pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.6;  // W/(m·K)
pub const WATER_SPECIFIC_HEAT: f64 = 4180.0;  // J/(kg·K)
pub const WATER_REFRACTIVE_INDEX: f64 = 1.33;
pub const WATER_GRUNEISEN: f64 = 0.12;

// Conversion factors
pub const MS_TO_S: f64 = 1e-3;
pub const MHZ_TO_HZ: f64 = 1e6;

// Default mode conversion efficiency
pub const DEFAULT_MODE_CONVERSION_EFFICIENCY: f64 = 0.3;

// Power law absorption
pub const DEFAULT_POWER_LAW_EXPONENT: f64 = 1.05;  // Typical for biological tissues