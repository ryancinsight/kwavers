//! Fundamental physical constants

use std::f64::consts::PI;

/// Speed of sound in water at 20°C (m/s)
pub const SOUND_SPEED_WATER: f64 = 1500.0;

/// Speed of sound in soft tissue (m/s)
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Speed of sound in air at 20°C (m/s)
pub const SOUND_SPEED_AIR: f64 = 343.0;

/// Density of water at 20°C (kg/m³)
pub const DENSITY_WATER: f64 = 1000.0;

/// Density of soft tissue (kg/m³)
pub const DENSITY_TISSUE: f64 = 1050.0;

/// Density of air at 20°C (kg/m³)
pub const DENSITY_AIR: f64 = 1.225;

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