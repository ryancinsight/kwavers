//! Configuration types for Burn-based 1D Wave Equation PINN
//!
//! Provides [`BurnPINNConfig`] and [`BurnLossWeights`] with validation and domain presets.
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks"
//!   Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045

mod pinn;
#[cfg(test)]
mod tests;
mod weights;

pub use pinn::BurnPINNConfig;
pub use weights::BurnLossWeights;
