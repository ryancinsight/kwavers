//! Multi-Modality Medical Image Fusion
//!
//! This module provides comprehensive support for registering and fusing multiple
//! medical imaging modalities (CT, MR, ultrasound) for improved diagnostics and therapy.
//!
//! ## Image Fusion Workflow
//!
//! ```text
//! Reference Image (CT/MR)
//!        ↓
//! Feature Extraction → Registration → Transform → Fusion → Output
//!        ↓                ↓             ↓         ↓
//!    Landmarks      Affine/Deformable Warp    Blending
//!    Edges          B-spline                   Overlay
//!    Intensity      Thin-plate spline          False color
//! ```
//!
//! ## Supported Modalities
//!
//! - **CT**: Dense 3D anatomical reference
//! - **MR**: High soft-tissue contrast
//! - **Ultrasound**: Real-time functional imaging
//! - **PET/SPECT**: Metabolic/functional data
//!
//! ## Registration Methods
//!
//! 1. **Rigid (6 DOF)**: Translation + rotation
//! 2. **Affine (12 DOF)**: Includes scaling + shear
//! 3. **Non-rigid**: Deformable registration for anatomical variations
//!
//! ## Fusion Output Modes
//!
//! - **Overlay**: Transparent overlay of floating image
//! - **Checkerboard**: Alternating tiles (alignment verification)
//! - **Difference**: Subtraction (change detection)
//! - **False Color**: Color-coded fusion
//!
//! ## References
//!
//! - Maintz & Viergever (1998): "A survey of medical image registration"
//! - Hill et al. (2001): "Medical image registration"
//! - Rueckert et al. (1999): "Non-rigid registration using free-form deformations"

pub mod fusion;
pub mod image;
pub mod manager;
pub mod parameters;
pub mod transform;

pub use fusion::FusionEngine;
pub use image::{ImageData, ImageModality};
pub use manager::{MultimodalityFusionManager, MultimodalitySession};
pub use parameters::{FusionParameters, MultimodalityFusionMethod};
pub use transform::{RegistrationTransform, TransformationType};
