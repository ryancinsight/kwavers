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

mod bessel;
mod solution;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use bessel::bessel_j;
pub use solution::{burgers_equation, fubini_harmonic_amplitude};
