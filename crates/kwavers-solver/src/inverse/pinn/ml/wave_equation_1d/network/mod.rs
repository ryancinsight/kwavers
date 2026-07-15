//! Neural network architecture for Coeus-backed 1D Wave Equation PINN.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::PinnWave1D;
