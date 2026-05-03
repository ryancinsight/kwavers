//! Single-Bubble Sonoluminescence (SBSL) Experimental Benchmarks
//!
//! Validates key analytical predictions of the Keller-Miksis bubble model
//! against the three canonical experimental datasets for SBSL:
//!
//! 1. **Brenner, Hilgenfeldt & Lohse (2002)** — comprehensive review with
//!    calibrated parameter sets for air bubbles in water at 26.5 kHz.
//! 2. **Yasui (1997)** — temperature-dependent light emission predictions.
//! 3. **Putterman & Weninger (2000)** — spectroscopic measurements.

pub mod conditions;
pub(super) mod constants;
pub mod kernels;
#[cfg(test)]
mod tests;

pub use conditions::BrennerSBSLConditions;
pub use kernels::{
    blake_threshold, collapse_time_fraction, minnaert_resonance_radius, planck_radiance_relative,
    wien_peak_wavelength_m, yasui_intensity_ratio,
};
