//! Acoustic Wave Physics Domain for PINN
//!
//! Implements acoustic wave equations for ultrasound physics using
//! Physics-Informed Neural Networks. Supports linear and nonlinear acoustic
//! wave propagation in homogeneous and heterogeneous media.

pub mod domain;
#[cfg(test)]
mod tests;
pub mod types;

pub use domain::AcousticWaveDomain;
pub use types::{AcousticBoundarySpec, AcousticBoundaryType, AcousticProblemType};
