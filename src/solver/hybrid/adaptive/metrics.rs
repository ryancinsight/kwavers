// adaptive/metrics.rs - Metrics for the adaptive module (compatibility layer)

use ndarray::Array3;

/// Detailed metrics for adaptive selection
#[derive(Debug, Clone)]
pub struct DetailedMetrics {
    pub smoothness: f64,
    pub discontinuities: usize,
    pub frequency_content: f64,
}

/// Quality metrics for method selection
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub accuracy_score: f64,
    pub efficiency_score: f64,
    pub stability_score: f64,
}

impl DetailedMetrics {
    pub fn compute(field: &Array3<f64>) -> Self {
        Self {
            smoothness: field.std(),
            discontinuities: 0,
            frequency_content: 0.5,
        }
    }
}

impl QualityMetrics {
    pub fn default() -> Self {
        Self {
            accuracy_score: 0.8,
            efficiency_score: 0.7,
            stability_score: 0.9,
        }
    }
}