// src/physics/effects/mechanical/mod.rs
//! Mechanical physics effects

mod damage;
mod erosion;

pub use damage::MaterialDamageEffect;
pub use erosion::ErosionEffect;