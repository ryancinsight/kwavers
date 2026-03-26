//! Physical constants for thermodynamics
//!
//! Re-exports authoritative values from `core::constants::fundamental`
//! plus thermodynamics-specific water properties.

pub use crate::core::constants::fundamental::AVOGADRO;
pub use crate::core::constants::fundamental::GAS_CONSTANT as R_GAS;
/// Molecular weight of water [kg/mol]
pub const M_WATER: f64 = 0.01801528;
/// Critical temperature of water \[K\]
pub const T_CRITICAL_WATER: f64 = 647.096;
/// Critical pressure of water \[Pa\]
pub const P_CRITICAL_WATER: f64 = 22.064e6;
/// Triple point temperature of water \[K\]
pub const T_TRIPLE_WATER: f64 = 273.16;
/// Triple point pressure of water \[Pa\]
pub const P_TRIPLE_WATER: f64 = 611.657;
/// Standard atmospheric pressure \[Pa\]
pub const P_ATM: f64 = 101325.0;
/// Enthalpy of vaporization for water at 100°C [J/mol]
pub const H_VAP_WATER_100C: f64 = 40660.0;
/// Boiling point of water at 1 atm \[K\]
pub const T_BOILING_WATER: f64 = 373.15;
