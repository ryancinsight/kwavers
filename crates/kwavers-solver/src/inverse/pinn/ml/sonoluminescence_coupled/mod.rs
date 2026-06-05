//! Sonoluminescence-Electromagnetic Coupled Physics Domain for PINN

mod config;
mod domain;

#[cfg(test)]
mod tests;

pub use config::{SonoluminescenceCouplingConfig, SonoluminescenceCouplingType};
pub use domain::SonoluminescenceCoupledDomain;
