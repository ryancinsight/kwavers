//! Hybrid Angular Spectrum (HAS) Method for Nonlinear Wave Propagation
//!
//! Implements efficient nonlinear propagation using operator splitting with
//! angular spectrum for diffraction and time-domain for nonlinearity.
//!
//! ## Literature References
//!
//! - Christopher, P. T., & Parker, K. J. (1991). JASA 90(1), 507-521.
//! - Zemp, R. J., et al. (2003). JASA 113(1), 139-152.
//! - Treeby, B. E., & Cox, B. T. (2010). JASA 127(5), 2741-2748.

pub mod absorption;
pub mod config;
pub mod diffraction;
pub mod facade;
pub mod nonlinearity;
pub mod solver;
#[cfg(test)]
mod tests;

pub use absorption::HasAbsorptionOperator;
pub use config::HASConfig;
pub use diffraction::HybridAsDiffractionOperator;
pub use facade::HybridAngularSpectrum;
pub use nonlinearity::HybridAsNonlinearOperator;
pub use solver::HybridAngularSpectrumSolver;
