//! Fundamental physical constants

use std::f64::consts::PI;

/// Speed of light in vacuum [m/s]
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Vacuum permittivity [F/m]
pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_817e-12;

/// Vacuum permeability [H/m]
pub const VACUUM_PERMEABILITY: f64 = 4.0 * PI * 1e-7;

/// Boltzmann constant [J/K]
pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23;

/// Avogadro's number [1/mol]
pub const AVOGADRO_NUMBER: f64 = 6.022_140_76e23;

/// Universal gas constant [J/(mol·K)]
pub const GAS_CONSTANT: f64 = 8.314_462_618;

/// Standard atmospheric pressure [Pa]
pub const STANDARD_PRESSURE: f64 = 101_325.0;

/// Standard temperature [K]
pub const STANDARD_TEMPERATURE: f64 = 293.15;

/// Gravitational acceleration [m/s²]
pub const GRAVITY: f64 = 9.80665;

/// Planck constant [J·s]
pub const PLANCK_CONSTANT: f64 = 6.626_070_15e-34;

/// Reduced Planck constant [J·s]
pub const HBAR: f64 = PLANCK_CONSTANT / (2.0 * PI);

/// Elementary charge [C]
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

/// Electron mass [kg]
pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;

/// Proton mass [kg]
pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;

/// Reference frequency for absorption [Hz]
pub const REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ: f64 = 1e6;

/// B/A divisor for nonlinearity
pub const B_OVER_A_DIVISOR: f64 = 2.0;

/// Nonlinearity coefficient offset
pub const NONLINEARITY_COEFFICIENT_OFFSET: f64 = 1.0;

/// Grid center factor
pub const GRID_CENTER_FACTOR: f64 = 0.5;