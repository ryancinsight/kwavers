//! Fundamental physical constants


/// Speed of sound in water at 20°C (m/s)
/// Value: 1482.0 m/s (more precise value)
/// Reference: National Physical Laboratory acoustic properties database
pub const SOUND_SPEED_WATER: f64 = 1482.0;

/// Speed of sound in soft tissue (m/s)
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Speed of sound in air at 20°C (m/s)
pub const SOUND_SPEED_AIR: f64 = 343.0;

/// Density of water at 20°C (kg/m³)
/// Value: 998.2 kg/m³ (precise value)
/// Reference: NIST Chemistry WebBook
pub const DENSITY_WATER: f64 = 998.2;

/// Density of soft tissue (kg/m³)
pub const DENSITY_TISSUE: f64 = 1050.0;

/// Density of air at 20°C (kg/m³)
/// Value: 1.204 kg/m³ (at 20°C, 1 atm)
/// Reference: NIST Standard Reference Database
pub const DENSITY_AIR: f64 = 1.204;

/// Standard atmospheric pressure (Pa)
pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// Vapor pressure of water at 20°C (Pa)
pub const VAPOR_PRESSURE_WATER_20C: f64 = 2339.0;

/// Gravitational acceleration (m/s²)
pub const GRAVITY: f64 = 9.80665;

/// Universal gas constant (J/(mol·K))
pub const GAS_CONSTANT: f64 = 8.314462618;

/// Avogadro's number (1/mol)
pub const AVOGADRO: f64 = 6.02214076e23;

/// Boltzmann constant (J/K)
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Planck constant (J·s)
pub const PLANCK: f64 = 6.62607015e-34;

/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f64 = 299792458.0;

/// Stefan-Boltzmann constant (W/(m²·K⁴))
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Elementary charge (C)
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

// Pi is already available through std::f64::consts::PI

// ============================================================================
// Elastic Constants
// ============================================================================

/// Bond transformation factor for anisotropic media
pub const BOND_TRANSFORM_FACTOR: f64 = 1.0;

/// Lamé to stiffness conversion factor
pub const LAME_TO_STIFFNESS_FACTOR: f64 = 1.0;

/// Symmetry tolerance for elastic tensors
pub const SYMMETRY_TOLERANCE: f64 = 1e-6;