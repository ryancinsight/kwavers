//! Physical constants for acoustic simulations
//!
//! This module provides a single source of truth (SSOT) for all physical constants
//! used throughout the simulation, eliminating magic numbers and ensuring consistency.

pub mod acoustic_parameters;
pub mod cavitation;
pub mod chemistry;
pub mod fundamental;
pub mod medical;
pub mod numerical;
pub mod optical;
pub mod thermodynamic;

// Re-export all constants for convenience
pub use acoustic_parameters::*;
pub use cavitation::*;
pub use chemistry::*;
pub use fundamental::*;
pub use medical::*;
pub use numerical::*;
pub use optical::*;
pub use thermodynamic::*;
