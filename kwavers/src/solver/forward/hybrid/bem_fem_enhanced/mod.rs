//! Enhanced BEM-FEM Coupling with Burton-Miller Formulation.
//!
//! Extends BEM-FEM coupling by integrating the Burton-Miller formulation to
//! eliminate spurious resonances and improve solution stability.
//!
//! ## References
//!
//! - Burton, A. J., & Miller, G. F. (1971). "The application of integral equation methods
//!   to the numerical solution of some exterior boundary value problems". Proceedings of
//!   the Royal Society of London. Series A, 323(1553), 201-210.

pub mod config;
pub mod solver;
#[cfg(test)]
mod tests;
pub mod types;

pub use config::EnhancedBemFemConfig;
pub use solver::EnhancedBemFemSolver;
pub use types::{InterfaceQuality, RefinementStep, ValidationResult};
