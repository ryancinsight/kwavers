//! Neural network architecture for the 2D elastic-wave PINN.
//!
//! This module keeps the public `ElasticPINN2D` facade stable while isolating
//! the coeus network implementation in `network`.

mod network;

pub use network::ElasticPINN2D;

#[cfg(test)]
mod tests;
