//! Tissue ablation model for thermal therapy
//!
//! Implements thermal ablation kinetics based on Arrhenius equation for
//! protein denaturation and tissue necrosis.

pub mod field;
pub mod kinetics;
pub mod state;

pub use field::AblationField;
pub use kinetics::AblationKinetics;
pub use state::AblationState;

#[cfg(test)]
mod tests;
