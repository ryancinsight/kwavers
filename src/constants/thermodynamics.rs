//! Thermodynamics constants

/// Avogadro's number [1/mol]
pub const AVOGADRO: f64 = 6.022_140_76e23;

/// Universal gas constant [J/(mol·K)]
pub const R_GAS: f64 = 8.314_462_618;

/// Molecular weight of water [kg/mol]
pub const M_WATER: f64 = 0.018_015;

/// Critical temperature of water [K]
pub const WATER_CRITICAL_TEMP: f64 = 647.096;

/// Critical pressure of water [Pa]
pub const WATER_CRITICAL_PRESSURE: f64 = 22.064e6;

/// Triple point temperature of water [K]
pub const WATER_TRIPLE_TEMP: f64 = 273.16;

/// Triple point pressure of water [Pa]
pub const WATER_TRIPLE_PRESSURE: f64 = 611.657;

/// Ambient temperature [K]
pub const T_AMBIENT: f64 = 293.15;

/// Nusselt constant
pub const NUSSELT_CONSTANT: f64 = 2.0;

/// Nusselt-Peclet coefficient
pub const NUSSELT_PECLET_COEFF: f64 = 0.6;

/// Nusselt-Peclet exponent
pub const NUSSELT_PECLET_EXPONENT: f64 = 0.5;

/// Sherwood-Peclet exponent
pub const SHERWOOD_PECLET_EXPONENT: f64 = 0.33;

/// Vapor diffusion coefficient [m²/s]
pub const VAPOR_DIFFUSION_COEFFICIENT: f64 = 2.5e-5;

/// Sonochemistry base rate
pub const SONOCHEMISTRY_BASE_RATE: f64 = 1e-6;

/// Reaction reference temperature [K]
pub const REACTION_REFERENCE_TEMPERATURE: f64 = 298.15;

/// Secondary reaction rate
pub const SECONDARY_REACTION_RATE: f64 = 1e-7;
