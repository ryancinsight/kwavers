//! Physics-informed neural network (PINN) beamforming processor.
//!
//! Implements beamforming with PINN-based delay calculation (eikonal equation),
//! adaptive weight computation, and Bayesian uncertainty quantification.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::NeuralBeamformingProcessor;
