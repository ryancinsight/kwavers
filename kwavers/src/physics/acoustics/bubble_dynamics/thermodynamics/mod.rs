//! Thermodynamic models for bubble dynamics
//!
//! This module provides comprehensive thermodynamic models for calculating
//! vapor pressure, phase equilibria, and thermal properties in bubble dynamics.
//!
//! # Models Implemented
//!
//! 1. **Antoine Equation**: Industry-standard for vapor pressure
//! 2. **Clausius-Clapeyron Relation**: Theoretical foundation for phase transitions
//! 3. **Wagner Equation**: High-accuracy model for water vapor
//! 4. **IAPWS-IF97**: International standard for water/steam properties
//!
//! # References
//!
//! - Wagner, W., & Pruss, A. (2002). "The IAPWS formulation 1995 for the
//!   thermodynamic properties of ordinary water substance." J. Phys. Chem. Ref. Data.
//! - Antoine, C. (1888). "Tensions des vapeurs; nouvelle relation entre les
//!   tensions et les températures." Comptes Rendus, 107, 681-684.

pub mod collapse;
pub mod constants;
pub mod mass_transfer;
pub mod thermal_properties;
pub mod vapor_pressure;

#[cfg(test)]
mod tests;

pub use collapse::calculate_collapse_temperature;
pub use mass_transfer::MassTransferModel;
pub use vapor_pressure::{ThermodynamicsCalculator, VaporPressureModel};
