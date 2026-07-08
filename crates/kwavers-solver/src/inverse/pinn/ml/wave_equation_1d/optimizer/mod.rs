//! Optimization components for Coeus-backed 1D Wave Equation PINN.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::SimpleOptimizer;
