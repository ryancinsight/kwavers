//! Wave propagation interfaces module

pub mod fresnel;
pub mod interface;
pub mod reflection;
pub mod refraction;
pub mod snell;

pub use fresnel::FresnelCalculator;
pub use interface::{Interface, InterfaceType};
pub use snell::SnellLawCalculator;

// Re-export Polarization from parent module for use by child modules
pub use super::Polarization;
