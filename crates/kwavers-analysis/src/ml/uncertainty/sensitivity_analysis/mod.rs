//! Variance-based global sensitivity analysis.

pub mod analyzer;
pub mod config;
#[cfg(test)]
mod tests;
pub mod types;

pub use analyzer::SensitivityAnalyzer;
pub use config::SensitivityConfig;
pub use types::{MorrisResults, SensitivityIndices};
