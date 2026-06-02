//! Spectral Unmixing for Photoacoustic Imaging
//!
//! Provides algorithms for decomposing multi-wavelength photoacoustic signals
//! into constituent chromophore concentrations.
//!
//! # Mathematical Foundation
//!
//! ## Linear Spectral Unmixing Model
//!
//! At each spatial location, the absorption coefficient at wavelength λᵢ is:
//!
//! ```text
//! μₐ(λᵢ) = Σⱼ εⱼ(λᵢ) · Cⱼ
//! ```
//!
//! In matrix form for M wavelengths and N chromophores:
//!
//! ```text
//! μ = E · C
//! ```
//!
//! Where:
//! - μ ∈ ℝᴹ: Measured absorption coefficients [μₐ(λ₁), ..., μₐ(λₘ)]ᵀ
//! - E ∈ ℝᴹˣᴺ: Extinction coefficient matrix (εⱼ(λᵢ))
//! - C ∈ ℝᴺ: Chromophore concentrations [C₁, ..., Cₙ]ᵀ

pub mod solvers;
pub mod types;

pub use solvers::unmixer::SpectralUnmixer;
pub use types::{SpectralUnmixingConfig, UnmixingResult, VolumetricUnmixingResult};
