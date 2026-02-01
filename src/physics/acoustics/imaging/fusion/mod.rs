//! Multi-Modal Imaging Fusion
//!
//! This module provides advanced fusion techniques for combining multiple imaging modalities
//! including ultrasound, photoacoustic imaging, and elastography. The fusion enables
//! comprehensive tissue characterization and improved diagnostic accuracy.
//!
//! ## Module Organization
//!
//! This module follows a deep vertical hierarchy with clear separation of concerns:
//!
//! - [`config`] - Configuration types and fusion parameters
//! - [`types`] - Core data structures and result types
//! - [`algorithms`] - Fusion algorithm implementations
//! - [`registration`] - Image registration and resampling
//! - [`quality`] - Quality assessment and uncertainty quantification
//! - [`properties`] - Tissue property extraction
//!
//! ## Fusion Techniques
//!
//! - **Spatial Registration**: Precise alignment of images from different modalities
//! - **Feature Fusion**: Combining complementary tissue properties
//! - **Probabilistic Fusion**: Uncertainty-aware combination of measurements
//! - **Deep Fusion**: Neural network-based multi-modal integration
//!
//! ## Clinical Benefits
//!
//! - **Enhanced Contrast**: Combining optical absorption (PA) with acoustic scattering (US)
//! - **Mechanical Properties**: Elastography provides tissue stiffness information
//! - **Molecular Imaging**: Photoacoustic enables functional and molecular contrast
//! - **Comprehensive Diagnosis**: Multi-parametric tissue assessment
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use kwavers::physics::acoustics::imaging::fusion::{FusionConfig, MultiModalFusion};
//!
//! // Create fusion processor with default configuration
//! let config = FusionConfig::default();
//! let mut fusion = MultiModalFusion::new(config);
//!
//! // Register multiple modalities
//! fusion.register_ultrasound(&ultrasound_data)?;
//! fusion.register_photoacoustic(&pa_result)?;
//! fusion.register_elastography(&elasticity_map)?;
//!
//! // Perform fusion
//! let fused_result = fusion.fuse()?;
//!
//! // Extract tissue properties
//! let properties = extract_tissue_properties(&fused_result);
//! ```
//!
//! ## Literature References
//!
//! - **Fused Imaging** (2020): "Multimodal imaging: A review of different fusion techniques"
//!   *Biomedical Optics Express*, 11(5), 2287-2305.
//!   DOI: [10.1364/BOE.388702](https://doi.org/10.1364/BOE.388702)
//!
//! - **Photoacoustic-Ultrasound** (2019): "Photoacoustic-ultrasound imaging fusion methods"
//!   *IEEE Transactions on Medical Imaging*, 38(9), 2023-2034.
//!   DOI: [10.1109/TMI.2019.2891290](https://doi.org/10.1109/TMI.2019.2891290)
//!
//! - **Image Registration** (2018): "Medical image registration: A review"
//!   *Medical Image Analysis*, 45, 1-26.
//!   DOI: [10.1016/j.media.2018.02.005](https://doi.org/10.1016/j.media.2018.02.005)
//!
//! ## Architectural Design
//!
//! This module implements Clean Architecture principles:
//!
//! - **Domain Layer**: Core types and business logic ([`types`], [`config`])
//! - **Application Layer**: Fusion algorithms and orchestration ([`algorithms`])
//! - **Infrastructure Layer**: Registration and quality assessment ([`registration`], [`quality`])
//! - **Interface Layer**: Public API through re-exports
//!
//! All dependencies flow inward toward the domain layer, ensuring testability
//! and maintainability.

// Module declarations
pub mod algorithms;
pub mod config;
pub mod properties;
pub mod quality;
pub mod registration;
pub mod types;

// Public re-exports for convenience
pub use algorithms::MultiModalFusion;
pub use config::RegistrationMethod;
pub use properties::extract_tissue_properties;

// Re-export domain types (moved to domain layer for clean architecture)
pub use crate::domain::imaging::fusion::{
    AffineTransform, FusedImageResult, FusionConfig, FusionMethod,
};
pub use types::RegisteredModality;

#[cfg(test)]
mod tests;
