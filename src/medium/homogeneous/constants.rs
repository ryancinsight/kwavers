//! Constants for homogeneous medium

/// Default optical properties for water at typical wavelengths

/// Default absorption coefficient for water [1/m]
/// Based on typical values for near-infrared wavelengths
pub const DEFAULT_WATER_ABSORPTION_COEFFICIENT: f64 = 0.1;

/// Default reduced scattering coefficient for water [1/m]
/// Based on typical values for biological tissue imaging
pub const DEFAULT_WATER_SCATTERING_COEFFICIENT: f64 = 1.0;

/// Default density for water [kg/m³]
pub const WATER_DENSITY: f64 = 1000.0;

/// Default sound speed in water [m/s]
pub const WATER_SOUND_SPEED: f64 = 1500.0;

/// Default nonlinearity parameter for water
pub const WATER_NONLINEARITY: f64 = 3.5;

/// Default attenuation coefficient for water [dB/(MHz^y cm)]
pub const WATER_ATTENUATION: f64 = 0.0022;

/// Default attenuation power law exponent for water
pub const WATER_ATTENUATION_POWER: f64 = 2.0;

/// Quantization factor for float comparison in caching
pub const FLOAT_QUANTIZATION_FACTOR: f64 = 1e6;
