//! Fast Nearfield Method (FNM) for efficient transducer field computation.

pub mod core;
#[cfg(test)]
mod tests;
pub mod types;

pub use core::FastNearfieldSolver;
pub use types::{AngularSpectrumFactors, FNMConfig};
