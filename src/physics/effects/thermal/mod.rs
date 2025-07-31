// src/physics/effects/thermal/mod.rs
//! Thermal physics effects

mod diffusion;
mod shock;

pub use diffusion::HeatDiffusionEffect;
pub use shock::ThermalShockEffect;