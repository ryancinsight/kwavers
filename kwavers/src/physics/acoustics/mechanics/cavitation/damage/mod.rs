//! Cavitation damage and erosion modeling
//!
//! Calculates mechanical damage from cavitation bubble collapse
//! including erosion, pitting, and material fatigue.

pub mod erosion;
pub mod material;
pub mod model;
pub mod bio_damage;

pub use erosion::{cavitation_intensity, ErosionPattern};
pub use material::{DamageParameters, MaterialProperties};
pub use model::CavitationDamage;

#[cfg(test)]
mod tests;
