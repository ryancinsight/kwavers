//! Type definitions for Coeus-backed 1D Wave Equation PINN

mod metrics;

#[cfg(test)]
mod tests;

pub use metrics::TrainingMetrics;
