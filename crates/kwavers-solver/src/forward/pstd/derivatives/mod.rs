//! Spectral Derivative Operators for Pseudospectral Methods
//!
//! Implements spectral (FFT-based) spatial derivative operators achieving
//! exponential convergence for smooth fields.
//!
//! # References
//!
//! - Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods
//! - Trefethen, L. N. (2000). Spectral Methods in MATLAB

pub mod operator;
#[cfg(test)]
mod tests;

pub use operator::SpectralDerivativeOperator;
