//! Optimization components for Burn-based 1D Wave Equation PINN.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::SimpleOptimizer;
