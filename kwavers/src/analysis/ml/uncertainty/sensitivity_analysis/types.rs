//! Output types for sensitivity analysis.

use std::collections::HashMap;

/// Sensitivity indices for each parameter
#[derive(Debug)]
pub struct SensitivityIndices {
    /// First-order sensitivity indices
    pub first_order: HashMap<String, f64>,
    /// Total sensitivity indices
    pub total: HashMap<String, f64>,
    /// Confidence intervals for indices
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Parameter rankings by sensitivity
    pub parameter_ranking: Vec<(String, f64)>,
}

/// Results from Morris screening
#[derive(Debug)]
pub struct MorrisResults {
    /// Mean elementary effects (μ)
    pub mu: Vec<f64>,
    /// Standard deviation of elementary effects (σ)
    pub sigma: Vec<f64>,
    /// All elementary effects
    pub elementary_effects: Vec<f64>,
}
