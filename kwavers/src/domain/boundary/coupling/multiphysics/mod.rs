//! Multi-physics interface boundary condition.
//!
//! Handles coupling between different physics domains with physically rigorous
//! transmission conditions.

pub mod interface;
#[cfg(test)]
mod tests;

pub use interface::MultiPhysicsInterface;
