//! Analytical Solutions for k-Wave Comparison and Validation
//!
//! Provides exact analytical solutions to the acoustic wave equation
//! for validation of numerical solvers against k-Wave and mathematical ground truth.
//!
//! # References
//!
//! 1. Pierce, A. D. (1989). *Acoustics*. Acoustical Society of America.
//! 2. Kinsler, L. E., et al. (2000). *Fundamentals of Acoustics* (4th ed.). Wiley.
//! 3. Goodman, J. W. (2005). *Introduction to Fourier Optics* (3rd ed.).
//! 4. Treeby, B. E., & Cox, B. T. (2010). k-Wave. *J. Biomed. Opt.*, 15(2), 021314.

mod error_metrics;
mod gaussian_beam;
mod plane_wave;
mod spherical_wave;
#[cfg(test)]
mod tests;

pub use error_metrics::ErrorMetrics;
pub use gaussian_beam::GaussianBeam;
pub use plane_wave::PlaneWave;
pub use spherical_wave::SphericalWave;
