//! Multi-Modal Image Registration for Spatial and Temporal Alignment
//!
//! This module provides comprehensive registration algorithms for aligning images
//! from different modalities (ultrasound, optical, photoacoustic, elastography).
//! Registration is critical for meaningful multi-modal fusion and accurate
//! tissue characterization.
//!
//! ## Registration Types
//!
//! - **Spatial Registration**: Aligns images in 2D/3D space
//! - **Temporal Registration**: Synchronizes acquisition timing
//! - **Modal Registration**: Aligns different imaging modalities
//!
//! ## Algorithms Implemented
//!
//! - **Rigid Body**: Translation + rotation (6 DOF in 3D)
//! - **Affine**: Linear transformation with scaling/shearing (12 DOF in 3D)
//! - **Feature-Based**: Landmark/feature matching and alignment
//! - **Intensity-Based**: Mutual information and correlation methods
//! - **Temporal**: Phase-locked acquisition synchronization
//!
//! ## Quality Metrics
//!
//! - **Fiducial Registration Error (FRE)**: Landmark alignment accuracy
//! - **Target Registration Error (TRE)**: Anatomical structure alignment
//! - **Mutual Information**: Statistical dependence measure
//! - **Correlation Coefficient**: Linear relationship measure
//! - **Temporal Jitter**: Acquisition timing synchronization
//!
//! ## Literature References
//!
//! - **Image Registration**: "Medical Image Registration" by Hajnal et al. (2001)
//! - **Multi-Modal Registration**: "Multi-modal image registration" by Sotiras et al. (2013)
//! - **Temporal Synchronization**: "Real-time multi-modal imaging" in IEEE TMI (2018)

pub mod engine;
pub mod intensity;
pub mod metrics;
pub mod spatial;
pub mod temporal;

#[cfg(test)]
mod tests;

// Re-export core types for easier API access
pub use engine::{ImageRegistration, RegistrationResult};
pub use metrics::{RegistrationQualityMetrics, TemporalQualityMetrics};
pub use spatial::SpatialTransform;
pub use temporal::TemporalSync;
