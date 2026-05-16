//! Acoustic wave mechanics module
//!
//! This module provides implementations for various acoustic wave propagation models
//! including linear and nonlinear wave equations.

pub mod nonlinear;
pub mod spatial_order;
pub mod wave_ops;

pub use nonlinear::NonlinearWave;
pub use spatial_order::SpatialOrder;
pub use wave_ops::{
    compute_diffusivity_from_power_law_absorption, compute_max_stable_timestep,
    compute_nonlinearity_coefficient,
};

#[cfg(test)]
mod tests;
