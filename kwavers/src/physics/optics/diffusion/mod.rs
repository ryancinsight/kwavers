//! Time-domain light diffusion solver
//!
//! Models the propagation of photons through highly scattering media using
//! the diffusion approximation to the radiative transport equation.

pub mod properties;
pub mod solver;

#[cfg(test)]
mod tests;

pub use properties::DiffusionOpticalProperties;
pub use solver::LightDiffusion;
