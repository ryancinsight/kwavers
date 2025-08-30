//! Material property constants
//!
//! Standard values for common materials used in acoustic simulations.

// Water properties at 20°C
pub const WATER_DENSITY: f64 = 1000.0; // kg/m³
pub const WATER_SOUND_SPEED: f64 = 1500.0; // m/s (approximate, varies with temperature)
pub const WATER_REFRACTIVE_INDEX: f64 = 1.333; // Optical, at 589 nm

// Air properties at 20°C, 1 atm
pub const AIR_DENSITY: f64 = 1.225; // kg/m³
pub const AIR_SOUND_SPEED: f64 = 343.0; // m/s

// Tissue properties (average soft tissue)
pub const TISSUE_DENSITY: f64 = 1050.0; // kg/m³
pub const TISSUE_SOUND_SPEED: f64 = 1540.0; // m/s
pub const TISSUE_NONLINEARITY: f64 = 3.5; // B/A parameter

// Bone properties
pub const BONE_DENSITY: f64 = 1900.0; // kg/m³
pub const BONE_SOUND_SPEED: f64 = 2800.0; // m/s

// Steel properties
pub const STEEL_DENSITY: f64 = 7850.0; // kg/m³
pub const STEEL_SOUND_SPEED: f64 = 5960.0; // m/s

// Absorption coefficients (dB/cm/MHz)
pub const WATER_ABSORPTION_COEFF: f64 = 0.0022;
pub const TISSUE_ABSORPTION_COEFF: f64 = 0.6;
pub const BONE_ABSORPTION_COEFF: f64 = 10.0;
