//! Skull-adaptive transcranial simulation benchmark.
//!
//! The benchmark follows the existing Chapter 25 brain pipeline: CT-derived
//! skull masks, transcranial focused-bowl coordinates, skull-path phase/amplitude
//! correction, and Rayleigh field synthesis. It adds the TFUScapes-style
//! evaluation contract: a CT volume, transducer surface coordinates, a
//! reference pressure field, and localization/amplitude metrics.

mod metrics;
mod pipeline;
mod placement;
mod types;

pub use metrics::evaluate_pressure_field;
pub use pipeline::run_skull_adaptive_transcranial_benchmark;
pub use types::{
    PressureFieldMetrics, SkullAdaptiveBenchmarkConfig, SkullAdaptiveBenchmarkResult,
    SkullAwareTransducerPlacement,
};
