//! Transcranial focused ultrasound therapy planning.
//!
//! Implements skull-path phase correction (HU-based ray tracing), Rayleigh-
//! Sommerfeld pressure field synthesis, acoustic observables (intensity, MI,
//! cavitation), GBM subspot raster planning, and BBB-opening Hill-function
//! dose model. All physics delegated to Rust; no Python-side computation.
//!
//! # Physical model references
//! - Rayleigh-Sommerfeld: Goodman, "Introduction to Fourier Optics", 4th ed.
//! - HU-to-acoustic mapping: Aubry et al., "Transcranial focused ultrasound
//!   using patient-specific skull models", J. Acoust. Soc. Am. 2003.
//! - BBB Hill model: McDannold et al., "Targeted disruption of the
//!   blood-brain barrier with focused ultrasound", Phys. Med. Biol. 2008.
//! - CEM43 / Pennes: delegated to `ThermalSimulation` binding.

mod bbb;
mod benchmark;
mod geometry;
mod observables;
mod pipeline;
mod pressure;
mod skull_ray;
mod subspot;
mod thermal;
pub(super) mod types;

pub use benchmark::{
    evaluate_pressure_field, run_skull_adaptive_transcranial_benchmark, PressureFieldMetrics,
    SkullAdaptiveBenchmarkConfig, SkullAdaptiveBenchmarkResult, SkullAwareTransducerPlacement,
};
pub use pipeline::run_transcranial_fus_planning;
pub use thermal::{transcranial_pennes_thermal_dose, TranscranialThermalResult};
pub use types::{TranscranialFusPlan, TranscranialFusPlanConfig};

#[cfg(test)]
mod tests;
