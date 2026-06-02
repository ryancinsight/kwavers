//! Encapsulated bubble physics models
//!
//! Provides the mathematical models for shell dynamics:
//! - Church (1995) linearized shell mechanics
//! - Marmottant (2005) viscoelastic nonlinear buckling models

pub mod church;
pub mod marmottant;

pub use church::ChurchModel;
pub use marmottant::MarmottantModel;

#[cfg(test)]
mod tests;
