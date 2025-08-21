//! Full Kuznetsov Equation Implementation for Nonlinear Acoustics
//!
//! This module implements the complete Kuznetsov equation, which provides the most
//! comprehensive model for nonlinear acoustic wave propagation in lossy media.
//!
//! # Physics Background
//!
//! The Kuznetsov equation is:
//! ```text
//! ∇²p - (1/c₀²)∂²p/∂t² = -(β/ρ₀c₀⁴)∂²p²/∂t² - (δ/c₀⁴)∂³p/∂t³ + F
//! ```
//!
//! Where:
//! - p: acoustic pressure
//! - c₀: small-signal sound speed
//! - β = 1 + B/2A: nonlinearity coefficient
//! - ρ₀: ambient density
//! - δ: acoustic diffusivity (related to absorption and dispersion)
//! - F: source terms
//!
//! ## Key Features:
//!
//! 1. **Full Nonlinearity**: Includes all second-order nonlinear terms
//! 2. **Acoustic Diffusivity**: Third-order time derivative for thermoviscous losses
//! 3. **Dispersion**: Proper handling of frequency-dependent absorption
//! 4. **Harmonic Generation**: Comprehensive modeling of harmonic buildup
//!
//! ## Module Organization:
//!
//! - `config`: Configuration structures and validation
//! - `solver`: Core Kuznetsov equation solver implementation
//! - `workspace`: Memory management and workspace structures
//! - `numerical`: Numerical methods and k-space corrections
//! - `nonlinear`: Nonlinear term computation
//! - `diffusion`: Acoustic diffusivity and absorption

// Submodules
pub mod config;
pub mod diffusion;
pub mod nonlinear;
pub mod numerical;
pub mod solver;
pub mod spectral;
pub mod workspace;

// Re-export main types
pub use config::{AcousticEquationMode, KuznetsovConfig};
pub use solver::KuznetsovWave;
pub use workspace::KuznetsovWorkspace;

#[cfg(test)]
mod tests;
