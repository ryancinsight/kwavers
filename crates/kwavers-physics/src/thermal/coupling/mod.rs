//! Thermal-acoustic coupling effects
//!
//! Implements bidirectional coupling between acoustic waves and thermal fields:
//! - Acoustic → Thermal: Viscous absorption generates heat
//! - Thermal → Acoustic: Temperature affects acoustic properties
//!
//! References:
//! - Hamilton & Blackstock (1998) "Nonlinear Acoustics"
//! - Nyborg (1988) "Acoustic streaming in ultrasonic therapy"
//! - ter Haar & Coussios (2007) "High intensity focused ultrasound"
//! - Zeqiri (2008) "Cavitation and ultrasonic surgery"

pub mod coefficients;
pub mod heating;
pub mod nonlinear;
pub mod solver;
pub mod streaming;

#[cfg(test)]
mod tests;

pub use coefficients::TemperatureCoefficients;
pub use heating::AcousticHeatingSource;
pub use nonlinear::NonlinearHeating;
pub use solver::ThermalAcousticCoupling;
pub use streaming::AcousticStreaming;
