//! Legacy scattering module - DEPRECATED
//! 
//! **DEPRECATION NOTICE**: This module is deprecated as of v2.17.0.
//! Please use `physics::wave_propagation::scattering` instead, which provides
//! a unified interface for all scattering phenomena.
//!
//! Migration guide:
//! - `scattering::acoustic::RayleighScattering` → `wave_propagation::ScatteringCalculator`
//! - `scattering::acoustic::mie` → `wave_propagation::ScatteringCalculator::mie_coefficients`
//! - `scattering::optic::RayleighOpticalScatteringModel` → `wave_propagation::ScatteringCalculator`

#[deprecated(since = "2.17.0", note = "Use physics::wave_propagation::scattering instead")]
pub mod acoustic;

#[deprecated(since = "2.17.0", note = "Use physics::wave_propagation::scattering instead")]
pub mod optic;

// Re-export for backward compatibility (will be removed in v3.0.0)
#[deprecated(since = "2.17.0", note = "Use wave_propagation::ScatteringCalculator instead")]
pub use acoustic::{RayleighScattering, compute_rayleigh_scattering, compute_mie_scattering};

use crate::physics::wave_propagation::scattering::{ScatteringCalculator, ScatteringRegime};

/// Migration helper: Convert old RayleighScattering to new ScatteringCalculator
#[deprecated(since = "2.17.0", note = "This is a migration helper, use ScatteringCalculator directly")]
pub fn migrate_rayleigh_to_unified(frequency: f64, wave_speed: f64) -> ScatteringCalculator {
    ScatteringCalculator::new(frequency, wave_speed)
}