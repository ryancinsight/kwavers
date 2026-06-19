//! Encapsulated bubble physics models.
//!
//! All models share the modified Rayleigh-Plesset balance via the
//! [`EncapsulatedShellModel`] trait (SSOT driver); each supplies only its
//! effective surface tension, equilibrium gas pressure, and shell stress:
//! - Church (1995) thin-shell shear-elastic mechanics (`[(R/R0)² − 1]`)
//! - Hoff (2000) thin-shell, linear-displacement elastic (`[1 − R0/R]`)
//! - Marmottant (2005) viscoelastic nonlinear buckling/rupture σ(R)
//! - Sarkar (2005) interfacial elasticity σ(R) + surface dilatational viscosity

pub mod church;
pub mod hoff;
pub mod marmottant;
pub mod sarkar;
pub mod shell_model;

pub use church::ChurchModel;
pub use hoff::HoffModel;
pub use marmottant::MarmottantModel;
pub use sarkar::SarkarModel;
pub use shell_model::EncapsulatedShellModel;

#[cfg(test)]
mod tests;
