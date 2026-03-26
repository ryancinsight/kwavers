//! Nonlinear Wave Propagation Module
//!
//! This module implements high-fidelity models for nonlinear acoustic wave propagation,
//! particularly relevant for High-Intensity Focused Ultrasound (HIFU), harmonic imaging,
//! and parametric arrays.
//!
//! # Mathematical Foundation
//!
//! ## Burgers' Equation (1D Nonlinear Propagation)
//!
//! The generalized Burgers' equation models the combined effects of nonlinearity,
//! diffraction, and thermoviscous attenuation:
//!
//! ```text
//! ∂P/∂z - (β/(ρ₀c₀³)) P ∂P/∂τ - (δ/(2c₀³)) ∂²P/∂τ² = 0
//! ```
//!
//! Where:
//! - P: Acoustic pressure
//! - z: Propagation coordinate
//! - τ = t - z/c₀: Retarded time
//! - β = 1 + B/2A: Coefficient of nonlinearity
//! - ρ₀, c₀: Ambient density and sound speed
//! - δ: Diffusivity of sound (attenuation)
//!
//! ## Khokhlov-Zabolotskaya-Kuznetsov (KZK) Equation
//!
//! The KZK equation extends Burgers' equation to include diffraction (transverse laplacian):
//!
//! ```text
//! ∂²P/∂z∂τ = (c₀/2) ∇²_⊥ P + (δ/(2c₀³)) ∂³P/∂τ³ + (β/(2ρ₀c₀³)) ∂²(P²)/∂τ²
//! ```
//!
//! # Architecture
//!
//! Submodules organize specific phenomenon into discrete traits and functions:
//! - `burgers`: Burgers' equation approximations and implementations
//! - `harmonics`: Harmonic generation (Tissue Harmonic Imaging)
//! - `kzk`: (Placeholder) Framework for 3D KZK solvers
//! - `parametric`: Parametric arrays (sum and difference frequency generation)
//! - `parameters`: Core properties determining nonlinear behavior
//! - `saturation`: Amplitude limiting and acoustic saturation
//! - `shock`: Shock wave formation and physics
//!
//! # Safety/Verification
//!
//! All algorithms are heavily bounded by mathematical domain checks (e.g., verifying
//! maximum theoretical pressure limits, shock formation constraints).

pub mod burgers;
pub mod harmonics;
pub mod kzk;
pub mod parametric;
pub mod parameters;
pub mod saturation;
pub mod shock;

#[cfg(test)]
mod tests;

pub use parameters::{NonlinearParameters, TissueHarmonicProperties};
