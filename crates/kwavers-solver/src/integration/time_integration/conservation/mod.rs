//! Conservation monitoring for multi-rate time integration
//!
//! This module tracks conservation of mass, momentum, energy, and angular momentum
//! to ensure physical accuracy of multi-physics simulations.

mod coupling;
mod monitor;
mod types;

#[cfg(test)]
mod tests;

pub use coupling::ConservativeCoupling;
pub use monitor::ConservationMonitor;
pub use types::{ConservationError, ConservationHistory, ConservedQuantities};
