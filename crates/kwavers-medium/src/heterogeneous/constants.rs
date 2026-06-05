//! Physical constants specific to heterogeneous media.
//!
//! `MIN_PHYSICAL_DENSITY` and `MIN_PHYSICAL_SOUND_SPEED` are the canonical
//! definitions that live in [`crate::core`]; this module only
//! carries constants not found there.

/// Minimum physical viscosity (Pa·s)
pub const MIN_PHYSICAL_VISCOSITY: f64 = 1e-6;

/// Minimum surface tension (N/m)
pub const MIN_SURFACE_TENSION: f64 = 1e-6;

/// Default polytropic index for air [-]
///
/// γ = 1.4 for diatomic ideal gas. SSOT: delegates to `HEAT_CAPACITY_RATIO_DIATOMIC`.
pub const DEFAULT_POLYTROPIC_INDEX: f64 =
    kwavers_core::constants::thermodynamic::HEAT_CAPACITY_RATIO_DIATOMIC;
