//! Thermal index safety calculation.
//!
//! The clinical facade exposes explicit TIS/TIB/TIC model selection and keeps
//! model reference power as input data. This prevents a hidden soft-tissue
//! default from being reused for bone or cranial applications.

pub mod calculator;
#[cfg(test)]
mod tests;
pub mod types;

pub use calculator::ThermalIndexCalculator;
pub use types::{ThermalIndexModel, ThermalIndexResult, ThermalIndexStatus};
