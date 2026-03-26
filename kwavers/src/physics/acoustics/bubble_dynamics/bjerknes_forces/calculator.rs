//! Core calculator for Bjerknes forces

use super::types::BjerknesConfig;

/// Bjerknes force calculator for bubble-bubble interactions
#[derive(Debug)]
pub struct BjerknesCalculator {
    pub(crate) config: BjerknesConfig,
}

impl BjerknesCalculator {
    /// Create new Bjerknes force calculator
    #[must_use]
    pub fn new(config: BjerknesConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> BjerknesConfig {
        self.config
    }
}
