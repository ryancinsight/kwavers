//! Burgers equation analytical Fubini-Blackstock solution.
//!
//! The lossless 1D Burgers equation in retarded-time form is
//!
//! ```text
//! dP/dz = (beta / (rho0 c0^3)) P dP/dtau
//! ```
//!
//! For a monochromatic source, the pre-shock solution is the Fubini series and
//! the post-shock asymptote is the Blackstock sawtooth coefficient. This module
//! keeps the public facade stable while separating special functions, analytical
//! coefficients, and tests into single-responsibility child modules.

mod solution;

#[cfg(test)]
mod tests;

/// Canonical Bessel Jₙ (workspace SSOT) re-exported for the Fubini harmonic
/// coefficients in `solution` and the harmonic-amplitude tests.
pub(crate) use kwavers_math::special::bessel::jn as bessel_j;
pub use solution::{burgers_equation, fubini_harmonic_amplitude};
