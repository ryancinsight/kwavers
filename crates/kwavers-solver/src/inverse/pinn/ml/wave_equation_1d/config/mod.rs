//! Configuration types for Coeus-backed 1D Wave Equation PINN
//!
//! Provides [`PinnConfig`] and [`LossWeights`] with validation and domain presets.
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks"
//!   Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045

mod pinn;
#[cfg(test)]
mod tests;
mod weights;

pub use pinn::PinnConfig;
pub use weights::LossWeights;
