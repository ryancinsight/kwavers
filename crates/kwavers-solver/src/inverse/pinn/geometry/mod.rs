//! PINN-Specific Geometry Extensions
//!
//! Extends domain-layer geometric abstractions with PINN-specific
//! functionality for collocation point sampling, interface condition handling,
//! and adaptive refinement strategies.
//!
//! # References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Karniadakis et al. (2021): "Physics-informed machine learning" - Nature Reviews Physics 3:422-440

mod adaptive;
mod interface;
mod sampling;
#[cfg(test)]
mod tests;

pub use adaptive::AdaptiveRefinement;
pub use interface::{MultiRegionDomain, PinnGeometryInterfaceCondition};
pub use sampling::{CollocationSampler, CollocationSamplingStrategy};
