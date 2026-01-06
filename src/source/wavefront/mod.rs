//! Wavefront source types
//!
//! This module contains source types that generate specific wavefront patterns
//! such as plane waves, Gaussian beams, spherical waves, etc.

pub mod bessel;
pub mod gaussian;
pub mod plane_wave;
pub mod spherical;

pub use bessel::{BesselBuilder, BesselConfig, BesselSource};
pub use gaussian::{GaussianBuilder, GaussianConfig, GaussianSource};
pub use plane_wave::{PlaneWaveBuilder, PlaneWaveConfig, PlaneWaveSource};
pub use spherical::{SphericalBuilder, SphericalConfig, SphericalSource, SphericalWaveType};
