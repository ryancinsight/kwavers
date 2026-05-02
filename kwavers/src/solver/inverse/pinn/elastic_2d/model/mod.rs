//! Neural network architecture for the 2D elastic-wave PINN.
//!
//! This module keeps the public `ElasticPINN2D` facade stable while isolating
//! the Burn network implementation in `network`.

mod network;

pub use network::ElasticPINN2D;

#[cfg(all(test, feature = "pinn"))]
mod tests;
