//! Shock Capturing for Spectral Discontinuous Galerkin Methods
//!
//! This module implements shock-capturing techniques for handling
//! discontinuities in spectral DG simulations, including:
//! - Artificial viscosity methods
//! - Sub-cell resolution techniques
//! - Hybrid spectral/finite-volume approaches
//!
//! # Theory
//!
//! When discontinuities (shocks) are present, spectral methods suffer from
//! Gibbs oscillations. This module provides several approaches to handle these.

pub mod detector;
pub mod limiter;
pub mod viscosity;

pub use detector::ShockDetector;
pub use limiter::WENOLimiter;
pub use viscosity::ArtificialViscosity;
