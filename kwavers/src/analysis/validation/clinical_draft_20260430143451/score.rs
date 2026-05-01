//! Clinical acceptability scoring kernel.
//!
//! Weighted blend of three components: minimum-metric satisfaction (50%
//! weight per metric), maximum-error inverse satisfaction (25% per error),
//! and binary safety-threshold compliance (25% per threshold). The total
//! is renormalised to 0–100 by the accumulated weight, with a hard 100.0
//! cap.

use std::collections::HashMap;

use super::{ClinicalRequirements, ClinicalValidator};

impl ClinicalValidator {
    /// Calculate overall clinical acceptability score
    pub(super) fn calculate_clinical_score(
        &self,
        metrics: &HashMap<String, f64>,
        reqs: &ClinicalRequirements,
    ) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        // Score minimum metrics (higher is better)
        for (metric_name, min_value) in &reqs.minimum_metrics {
            if let Some(actual_value) = metrics.get(metric_name) {
                let score = (actual_value / min_value).min(2.0) * 50.0; // Cap at 200% of minimum
                total_score += score;
                total_weight += 50.0;
            }
        }

        // Score maximum errors (lower is better)
        for (error_name, max_error) in &reqs.maximum_errors {
            if let Some(actual_error) = metrics.get(error_name) {
                let error_ratio = (actual_error / max_error).min(2.0);
                let score = (2.0 - error_ratio) * 25.0; // Perfect score when error = 0
                total_score += score;
                total_weight += 25.0;
            }
        }

        // Score safety (binary)
        for (safety_name, max_value) in &reqs.safety_thresholds {
            if let Some(actual_value) = metrics.get(safety_name) {
                let score = if *actual_value <= *max_value {
                    25.0
                } else {
                    0.0
                };
                total_score += score;
                total_weight += 25.0;
            }
        }

        if total_weight > 0.0 {
            (total_score / total_weight * 100.0).min(100.0)
        } else {
            0.0
        }
    }
}
