//! Acoustic and ultrasound constants

/// Water properties at 20°C
pub const WATER_SOUND_SPEED: f64 = 1482.0; // [m/s]
pub const WATER_DENSITY: f64 = 998.2; // [kg/m³]
pub const WATER_NONLINEARITY: f64 = 5.2; // B/A parameter
pub const WATER_ATTENUATION: f64 = 0.0022; // [dB/(MHz²·cm)]
pub const WATER_SPECIFIC_HEAT: f64 = 4182.0; // [J/(kg·K)]
pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.598; // [W/(m·K)]

/// Air properties at 20°C
pub const AIR_SOUND_SPEED: f64 = 343.0; // [m/s]
pub const AIR_DENSITY: f64 = 1.204; // [kg/m³]
pub const AIR_NONLINEARITY: f64 = 0.4; // B/A parameter
pub const AIR_ATTENUATION: f64 = 1.64; // [dB/(MHz²·m)]

/// Soft tissue average properties
pub const TISSUE_SOUND_SPEED: f64 = 1540.0; // [m/s]
pub const TISSUE_DENSITY: f64 = 1050.0; // [kg/m³]
pub const TISSUE_NONLINEARITY: f64 = 6.8; // B/A parameter
pub const TISSUE_ATTENUATION: f64 = 0.75; // [dB/(MHz·cm)]

/// Bone properties
pub const BONE_SOUND_SPEED: f64 = 3500.0; // [m/s]
pub const BONE_DENSITY: f64 = 1900.0; // [kg/m³]
pub const BONE_NONLINEARITY: f64 = 8.0; // B/A parameter
pub const BONE_ATTENUATION: f64 = 20.0; // [dB/(MHz·cm)]

/// Medical ultrasound frequencies
pub const DIAGNOSTIC_FREQ_MIN: f64 = 1e6; // 1 MHz
pub const DIAGNOSTIC_FREQ_MAX: f64 = 20e6; // 20 MHz
pub const THERAPEUTIC_FREQ_MIN: f64 = 0.5e6; // 0.5 MHz
pub const THERAPEUTIC_FREQ_MAX: f64 = 5e6; // 5 MHz

/// HIFU parameters
pub const HIFU_FREQUENCY: f64 = 1e6; // 1 MHz typical
pub const HIFU_INTENSITY_MIN: f64 = 100.0; // W/cm²
pub const HIFU_INTENSITY_MAX: f64 = 10000.0; // W/cm²

/// Acoustic impedance = density * sound_speed
pub const WATER_IMPEDANCE: f64 = WATER_DENSITY * WATER_SOUND_SPEED;
pub const AIR_IMPEDANCE: f64 = AIR_DENSITY * AIR_SOUND_SPEED;
pub const TISSUE_IMPEDANCE: f64 = TISSUE_DENSITY * TISSUE_SOUND_SPEED;
pub const BONE_IMPEDANCE: f64 = BONE_DENSITY * BONE_SOUND_SPEED;