//! Conservation Violation Detectors
//!
//! Real-time detection and reporting of conservation law violations during simulation.

use super::checkers::{ConservationLaw, ConservationResult};
use crate::core::error::KwaversResult;
use std::collections::HashMap;

/// A detected conservation violation
#[derive(Debug, Clone)]
pub struct ConservationViolation {
    /// Timestep where violation occurred
    pub timestep: u64,

    /// Simulation time
    pub time: f64,

    /// Conserved quantity name (e.g., "pressure", "energy")
    pub quantity: String,

    /// Conservation law type
    pub law: ConservationLaw,

    /// Relative error observed
    pub relative_error: f64,

    /// Threshold that was exceeded
    pub threshold: f64,

    /// Severity level (0.0-1.0, where 1.0 is catastrophic)
    pub severity: f64,
}

/// Detects and analyzes conservation violations
#[derive(Debug)]
pub struct ConservationViolationDetector {
    /// Violations detected so far
    violations: Vec<ConservationViolation>,

    /// Error threshold for violation reporting
    error_threshold: f64,

    /// Violations per conservation law
    law_violations: HashMap<ConservationLaw, usize>,

    /// Current timestep
    timestep: u64,

    /// Maximum allowed violations before warning
    max_violations_before_warning: usize,
}

impl ConservationViolationDetector {
    /// Create a new violation detector
    ///
    /// # Arguments
    ///
    /// * `error_threshold` - Relative error threshold for violation (e.g., 1e-3)
    pub fn new(error_threshold: f64) -> Self {
        Self {
            violations: Vec::new(),
            error_threshold,
            law_violations: HashMap::new(),
            timestep: 0,
            max_violations_before_warning: 10,
        }
    }

    /// Process conservation check results and detect violations
    ///
    /// # Arguments
    ///
    /// * `results` - Conservation check results from ConservationChecker
    /// * `time` - Current simulation time
    pub fn detect(
        &mut self,
        results: &HashMap<String, ConservationResult>,
        time: f64,
    ) -> KwaversResult<Vec<ConservationViolation>> {
        let mut detected = Vec::new();

        for (name, result) in results {
            if !result.passed && result.relative_error > self.error_threshold {
                // Calculate severity as normalized log of error ratio
                let error_ratio = result.relative_error / self.error_threshold;
                let severity = (error_ratio.ln() + 1.0).min(1.0).max(0.0);

                let violation = ConservationViolation {
                    timestep: self.timestep,
                    time,
                    quantity: name.clone(),
                    law: result.law,
                    relative_error: result.relative_error,
                    threshold: self.error_threshold,
                    severity,
                };

                *self.law_violations.entry(result.law).or_insert(0) += 1;
                detected.push(violation.clone());
                self.violations.push(violation);
            }
        }

        self.timestep += 1;

        Ok(detected)
    }

    /// Get all violations recorded so far
    pub fn all_violations(&self) -> &[ConservationViolation] {
        &self.violations
    }

    /// Get violations for a specific conservation law
    pub fn violations_for_law(&self, law: ConservationLaw) -> Vec<&ConservationViolation> {
        self.violations.iter().filter(|v| v.law == law).collect()
    }

    /// Get most recent violations (last N)
    pub fn recent_violations(&self, n: usize) -> Vec<&ConservationViolation> {
        self.violations.iter().rev().take(n).collect()
    }

    /// Check if violations are becoming worse (trending upward)
    pub fn is_trend_worsening(&self, window_size: usize) -> bool {
        if self.violations.len() < window_size * 2 {
            return false;
        }

        let recent = self.violations.len() - window_size;
        let recent_avg = self.violations[recent..]
            .iter()
            .map(|v| v.relative_error)
            .sum::<f64>()
            / window_size as f64;

        let previous_avg = self.violations[recent - window_size..recent]
            .iter()
            .map(|v| v.relative_error)
            .sum::<f64>()
            / window_size as f64;

        recent_avg > previous_avg * 1.1 // 10% increase threshold
    }

    /// Get summary statistics
    pub fn statistics(&self) -> ViolationStatistics {
        let total_violations = self.violations.len();

        let max_error = self
            .violations
            .iter()
            .map(|v| v.relative_error)
            .fold(0.0, f64::max);

        let avg_error = if total_violations > 0 {
            self.violations
                .iter()
                .map(|v| v.relative_error)
                .sum::<f64>()
                / total_violations as f64
        } else {
            0.0
        };

        let critical_violations = self.violations.iter().filter(|v| v.severity > 0.8).count();

        ViolationStatistics {
            total_violations,
            critical_violations,
            max_relative_error: max_error,
            average_relative_error: avg_error,
            violations_per_law: self.law_violations.clone(),
        }
    }

    /// Clear violation history
    pub fn reset(&mut self) {
        self.violations.clear();
        self.law_violations.clear();
        self.timestep = 0;
    }
}

/// Statistics about conservation violations
#[derive(Debug, Clone)]
pub struct ViolationStatistics {
    /// Total number of violations detected
    pub total_violations: usize,

    /// Number of critical violations (severity > 0.8)
    pub critical_violations: usize,

    /// Maximum relative error observed
    pub max_relative_error: f64,

    /// Average relative error
    pub average_relative_error: f64,

    /// Violation count per conservation law
    pub violations_per_law: HashMap<ConservationLaw, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let detector = ConservationViolationDetector::new(1e-3);
        assert_eq!(detector.all_violations().len(), 0);
    }

    #[test]
    fn test_violation_detection() -> KwaversResult<()> {
        let mut detector = ConservationViolationDetector::new(1e-3);

        let mut results = HashMap::new();
        results.insert(
            "pressure".to_string(),
            ConservationResult {
                law: ConservationLaw::Mass,
                initial_value: 1.0,
                current_value: 1.01,
                absolute_change: 0.01,
                relative_error: 0.01,
                tolerance: 1e-10,
                passed: false,
                error_message: Some("Test violation".to_string()),
            },
        );

        let violations = detector.detect(&results, 0.0)?;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].relative_error, 0.01);

        Ok(())
    }

    #[test]
    fn test_severity_calculation() -> KwaversResult<()> {
        let mut detector = ConservationViolationDetector::new(1e-2);

        let mut results = HashMap::new();
        results.insert(
            "energy".to_string(),
            ConservationResult {
                law: ConservationLaw::Energy,
                initial_value: 100.0,
                current_value: 101.0,
                absolute_change: 1.0,
                relative_error: 0.1, // 10x threshold
                tolerance: 1e-10,
                passed: false,
                error_message: None,
            },
        );

        let violations = detector.detect(&results, 0.0)?;
        assert!(!violations.is_empty());
        assert!(violations[0].severity > 0.0);
        assert!(violations[0].severity <= 1.0);

        Ok(())
    }

    #[test]
    fn test_statistics() -> KwaversResult<()> {
        let mut detector = ConservationViolationDetector::new(1e-3);

        let mut results = HashMap::new();
        results.insert(
            "mass".to_string(),
            ConservationResult {
                law: ConservationLaw::Mass,
                initial_value: 1.0,
                current_value: 1.005,
                absolute_change: 0.005,
                relative_error: 0.005,
                tolerance: 1e-10,
                passed: false,
                error_message: None,
            },
        );

        detector.detect(&results, 0.0)?;

        let stats = detector.statistics();
        assert_eq!(stats.total_violations, 1);
        assert!(stats.max_relative_error > 0.0);

        Ok(())
    }

    #[test]
    fn test_violation_filtering() -> KwaversResult<()> {
        let mut detector = ConservationViolationDetector::new(1e-3);

        // Add multiple violations
        for i in 0..3 {
            let mut results = HashMap::new();
            results.insert(
                "pressure".to_string(),
                ConservationResult {
                    law: ConservationLaw::Mass,
                    initial_value: 1.0,
                    current_value: 1.0 + 0.005 * (i as f64 + 1.0),
                    absolute_change: 0.005 * (i as f64 + 1.0),
                    relative_error: 0.005 * (i as f64 + 1.0),
                    tolerance: 1e-10,
                    passed: false,
                    error_message: None,
                },
            );
            detector.detect(&results, i as f64 * 0.1)?;
        }

        let mass_violations = detector.violations_for_law(ConservationLaw::Mass);
        assert_eq!(mass_violations.len(), 3);

        let recent = detector.recent_violations(2);
        assert_eq!(recent.len(), 2);

        Ok(())
    }
}
