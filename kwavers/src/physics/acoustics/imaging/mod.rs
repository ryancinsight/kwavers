//! Medical Imaging Modalities for Acoustic Applications
//!
//! Provides image fusion, multi-modal processing, and registration algorithms
//! for medical ultrasound, photoacoustic imaging, and elastography applications.
//!
//! ## Submodules
//!
//! - **fusion**: Multi-modal image fusion and integration
//! - **modalities**: CEUS (Contrast-Enhanced Ultrasound), elastography, ultrasound
//! - **registration**: Spatial and temporal image registration
//!
//! ## Design Note
//!
//! Submodules are kept public for direct access. Users can import specific types:
//! ```ignore
//! use crate::physics::acoustics::imaging::fusion::MultiModalFusion;
//! use crate::physics::acoustics::imaging::registration::ImageRegistration;
//! ```
//! This eliminates namespace pollution from wildcard re-exports while maintaining
//! organized, hierarchical access to imaging functionality.

pub mod fusion;
pub mod modalities;
pub mod registration;

// NOTE: Wildcard re-exports removed to prevent namespace pollution.
// Access specific types via: physics::acoustics::imaging::submodule::Type
