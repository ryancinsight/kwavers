// src/physics/effects/optical/mod.rs
//! Optical physics effects
//! 
//! This module contains effects related to light emission, absorption,
//! and propagation in the acoustic field.

mod sonoluminescence;
mod photoacoustic;
mod light_diffusion;

pub use sonoluminescence::SonoluminescenceEffect;
pub use photoacoustic::PhotoacousticEffect;
pub use light_diffusion::LightDiffusionEffect;