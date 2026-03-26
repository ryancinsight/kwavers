//! Microbubble Cloud Dynamics for Contrast-Enhanced Ultrasound
//!
//! Simulates the collective behavior of microbubble populations in acoustic fields,
//! including bubble-bubble interactions, dissolution, coalescence, and nonlinear scattering.
//!
//! ## References
//!
//! - Church (1995) JASA 97(3):1510-1521
//! - Tang & Eckersley (2006) IEEE TUFFC 53(1):126-141
//! - Doinikov (2001) Phys. Fluids 13(8):2219-2226

pub mod config;
pub mod incident_field;
pub mod interactions;
pub mod scattering;
pub mod simulator;

#[cfg(test)]
mod tests;

pub use config::{CloudBubble, CloudConfig};
pub use incident_field::{CloudResponse, CloudState, IncidentField};
pub use scattering::ScatteredField;
pub use simulator::CloudDynamics;
