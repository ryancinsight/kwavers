//! Physical constants for acoustic simulations
//!
//! This module provides a single source of truth (SSOT) for all physical constants
//! used throughout the simulation, eliminating magic numbers and ensuring consistency.

pub mod acoustic_parameters;
pub mod fundamental;
pub mod thermodynamic;
pub mod optical;
pub mod cavitation;
pub mod numerical;
pub mod medical;
pub mod chemistry;

// Re-export all constants for convenience
pub use acoustic_parameters::*;
pub use fundamental::*;
pub use thermodynamic::*;
pub use optical::*;
pub use cavitation::*;
pub use numerical::*;
pub use medical::*;
pub use chemistry::*;