//! Neural network architecture for 3D wave equation PINN.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::PINN3DNetwork;
