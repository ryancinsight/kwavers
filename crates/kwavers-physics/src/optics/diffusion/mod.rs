//! Time-domain light diffusion solver
//!
//! Models the propagation of photons through highly scattering media using
//! the diffusion approximation to the radiative transport equation.

pub mod solver;

#[cfg(test)]
mod tests;

pub use solver::LightDiffusion;
