//! Clinical Decision Support for Neural Ultrasound Analysis
//!
//! Implements automated clinical analysis including lesion detection,
//! tissue classification, and diagnostic recommendations for ultrasound imaging.

use super::types::ClinicalThresholds;

mod analysis;
mod detection;
#[cfg(test)]
mod tests;

/// Clinical Decision Support System
///
/// Provides automated lesion detection, tissue classification, and diagnostic
/// recommendations based on neural network-enhanced ultrasound analysis.
#[derive(Debug, Clone)]
pub struct NeuralClinicalDecisionSupport {
    pub(super) config: ClinicalThresholds,
}

impl NeuralClinicalDecisionSupport {
    /// Create new clinical decision support system
    #[must_use]
    pub fn new(thresholds: ClinicalThresholds) -> Self {
        Self { config: thresholds }
    }
}
