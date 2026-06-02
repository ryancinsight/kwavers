//! Type definitions for Burn-based 1D Wave Equation PINN

mod metrics;

#[cfg(test)]
mod tests;

pub use metrics::BurnTrainingMetrics;
