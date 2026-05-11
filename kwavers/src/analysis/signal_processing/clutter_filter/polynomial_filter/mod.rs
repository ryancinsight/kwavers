//! Polynomial Regression Clutter Filter
//!
//! This module implements polynomial regression-based clutter filtering for ultrasound
//! Doppler imaging. The filter removes slow-moving tissue clutter by fitting and subtracting
//! a polynomial curve from the temporal signal at each spatial location.
//!
//! # Algorithm Overview
//!
//! For each pixel time series x(t), the filter:
//! 1. Fits a polynomial p(t) = Σᵢ aᵢtⁱ to the signal
//! 2. Subtracts the fitted polynomial: y(t) = x(t) - p(t)
//! 3. Returns the residual containing blood flow signal
//!
//! # Mathematical Foundation
//!
//! Polynomial regression solves the least-squares problem:
//! ```text
//! min Σₜ [x(t) - Σᵢ aᵢtⁱ]²
//! ```
//!
//! This is equivalent to solving the normal equations:
//! ```text
//! (VᵀV)a = Vᵀx
//! ```
//! where V is the Vandermonde matrix with elements Vₜᵢ = tⁱ
//!
//! # References
//!
//! - Jensen, J. A. (1996). *Field: A Program for Simulating Ultrasound Systems*
//! - Bjaerum, S., Torp, H., & Kristoffersen, K. (2002). "Clutter filters adapted to tissue motion in ultrasound color flow imaging"
//!   *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 49(6), 693-704.
//! - Yu, A. C. H., & Cobbold, R. S. C. (2008). "Single-ensemble-based eigen-processing methods for color flow imaging"
//!   *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 55(3), 559-572.

mod config;
mod filter;
#[cfg(test)]
mod tests;

pub use config::PolynomialFilterConfig;
pub use filter::PolynomialFilter;
