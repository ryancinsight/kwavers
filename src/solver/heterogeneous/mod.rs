//! Heterogeneous media solver components
//!
//! This module provides specialized handling for wave propagation in heterogeneous media
//! with proper domain separation following SOLID principles.

pub mod config;
pub mod handler;
pub mod smoothing;
pub mod splitting;
pub mod density_scaling;
pub mod kspace_correction;

pub use config::HeterogeneousConfig;
pub use handler::HeterogeneousHandler;
pub use smoothing::SmoothingMethod;
pub use splitting::PressureVelocitySplit;
pub use density_scaling::DensityScaler;
pub use kspace_correction::KSpaceCorrector;