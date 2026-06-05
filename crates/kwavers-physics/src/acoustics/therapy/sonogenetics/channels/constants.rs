//! Physical constants for sonogenetic channel gating.
//!
//! All numeric constants are sourced from the SSOT in `kwavers_core::constants`
//! to prevent independent duplicates from drifting apart.

/// Boltzmann constant [J/K] — re-exported from `kwavers_core::constants::fundamental`.
pub(super) use kwavers_core::constants::fundamental::BOLTZMANN as K_B;

// Note: use `kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K` directly
// at call sites. The BODY_TEMP_K alias has been removed to enforce SSOT.
