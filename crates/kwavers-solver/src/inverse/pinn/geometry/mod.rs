//! PINN-Specific Geometry Extensions
//!
//! Extends domain-layer geometric abstractions with PINN-specific
//! functionality for collocation point sampling and interface conditions.
//!
//! # References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Karniadakis et al. (2021): "Physics-informed machine learning" - Nature Reviews Physics 3:422-440

mod interface;
mod sampling;
#[cfg(test)]
mod tests;

pub use interface::{MultiRegionDomain, MultiRegionError, PinnGeometryInterfaceCondition};
pub use sampling::{CollocationSampler, CollocationSamplingStrategy};
