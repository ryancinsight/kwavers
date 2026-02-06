//! Anisotropic material support for biological tissues
//!
//! Implements full anisotropic material models including orthotropic,
//! transversely isotropic, and general anisotropic materials.
//!
//! # Architecture
//! - Modular design with clear separation of concerns
//! - Stiffness tensor operations separated from wave propagation
//! - Rotation and transformation utilities isolated
//!
//! # References
//! - Royer & Dieulesaint (2000). "Elastic waves in solids I" Springer.
//! - Aristizabal et al. (2018). "Shear wave vibrometry in ex vivo porcine lens" J Biomech

pub mod christoffel;
pub mod fiber;
pub mod rotation;
pub mod stiffness;
pub mod types;

pub use christoffel::ChristoffelEquation;
pub use fiber::{FiberOrientation, MuscleFiberModel};
pub use rotation::RotationMatrix;
pub use stiffness::StiffnessTensor;
pub use types::AnisotropyType;
