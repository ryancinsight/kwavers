//! Neural network architecture for Burn-based 1D Wave Equation PINN.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::BurnPINN1DWave;
