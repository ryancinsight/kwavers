//! # Spectral Operators
//!
//! FFT-based differential and filtering operators for pseudospectral
//! time-domain (PSTD) methods.
//!
//! ## Mathematical Foundation
//!
//! The spectral derivative uses the Fourier differentiation theorem:
//!
//! ```text
//! ∂u/∂x = F⁻¹{ik_x F{u}}
//! ```
//!
//! ## References
//!
//! - Liu, Q. H. (1997). Microwave Opt. Technol. Lett., 15(3), 158-165.
//! - Canuto et al. (2007). Spectral Methods: Fundamentals in Single Domains.

mod derivative;
mod filter;
#[cfg(test)]
mod tests;
mod trait_def;

pub use derivative::PseudospectralDerivative;
pub use filter::{FilterType, SpectralFilter};
pub use trait_def::SpectralOperator;
