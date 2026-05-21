//! Physical constants for sonogenetic channel gating.
//!
//! All numeric constants are sourced from the SSOT in `crate::core::constants`
//! to prevent independent duplicates from drifting apart.

/// Boltzmann constant [J/K] — re-exported from `crate::core::constants::fundamental`.
pub(super) use crate::core::constants::fundamental::BOLTZMANN as K_B;

/// Canonical body temperature (K) = 37 °C — re-exported from
/// `crate::core::constants::thermodynamic`.
pub use crate::core::constants::thermodynamic::BODY_TEMPERATURE_K as BODY_TEMP_K;
