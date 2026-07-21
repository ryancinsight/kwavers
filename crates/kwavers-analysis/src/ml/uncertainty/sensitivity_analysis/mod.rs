//! Reproducible correlation-based global sensitivity screening.

pub mod analyzer;
pub mod config;
#[cfg(test)]
mod tests;

pub use analyzer::SensitivityAnalyzer;
pub use config::SensitivityConfig;
pub use tyche_core::{Parameter, ParameterSpace, Seed, SensitivityReport};
