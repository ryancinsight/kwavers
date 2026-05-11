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

#[cfg(test)]
mod tests {
    use super::*;

    /// BjerknesCalculator::new stores the provided config.
    #[test]
    fn new_stores_config_fields() {
        let mut cfg = BjerknesConfig::default();
        cfg.frequency = 500e3;
        let calc = BjerknesCalculator::new(cfg);
        assert!((calc.config().frequency - 500e3).abs() < 1e-6,
            "frequency not stored: {}", calc.config().frequency);
    }

    /// Debug output is non-empty.
    #[test]
    fn debug_non_empty() {
        let calc = BjerknesCalculator::new(BjerknesConfig::default());
        assert!(!format!("{calc:?}").is_empty());
    }
}
