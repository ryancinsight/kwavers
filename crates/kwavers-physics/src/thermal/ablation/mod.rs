//! Tissue ablation model for thermal therapy
//!
//! Implements thermal ablation kinetics based on Arrhenius equation for
//! protein denaturation and tissue necrosis.

/// Spatial ablation field — per-voxel thermal-damage accumulation.
pub mod field;
pub mod kinetics;
/// Ablation state value objects (denatured fraction / necrosis lifecycle).
pub mod state;

pub use field::AblationField;
pub use kinetics::AblationKinetics;
pub use state::AblationState;

#[cfg(test)]
mod tests;
