//! SafetyController implementation.

use super::types::TherapyAction;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

use super::super::config::TherapyIntegrationSafetyLimits;
use super::super::state::{SafetyMetrics, TherapyIntegrationSafetyStatus};

/// Real-time safety controller for therapeutic ultrasound.
///
/// Monitors safety metrics and enforces limits during therapy execution.
#[derive(Debug, Clone)]
pub struct SafetyController {
    /// Configured safety limits
    limits: TherapyIntegrationSafetyLimits,
    /// Warning margins (fraction of limit, typically 0.8)
    warning_margins: HashMap<String, f64>,
    /// Current safety metrics
    current_metrics: SafetyMetrics,
    /// Per-organ accumulated dose
    organ_doses: HashMap<String, f64>,
    /// Maximum allowed dose per organ
    organ_max_doses: HashMap<String, f64>,
    /// Treatment start time (seconds)
    treatment_start_time: f64,
    /// Whether a violation has occurred (sticky flag)
    violation_detected: bool,
    /// Last action taken
    pub(super) last_action: TherapyAction,
    /// Number of warning events
    warning_count: u32,
    /// Number of power reduction events
    reduction_count: u32,
}

impl SafetyController {
    /// Create new safety controller.
    pub fn new(
        limits: TherapyIntegrationSafetyLimits,
        organ_limits: Option<HashMap<String, f64>>,
    ) -> Self {
        let mut warning_margins = HashMap::new();
        warning_margins.insert("thermal_index".to_string(), 0.8);
        warning_margins.insert("mechanical_index".to_string(), 0.8);
        warning_margins.insert("cavitation_dose".to_string(), 0.8);
        warning_margins.insert("treatment_time".to_string(), 0.9);

        Self {
            limits,
            warning_margins,
            current_metrics: SafetyMetrics::default(),
            organ_doses: HashMap::new(),
            organ_max_doses: organ_limits.unwrap_or_default(),
            treatment_start_time: 0.0,
            violation_detected: false,
            last_action: TherapyAction::Continue,
            warning_count: 0,
            reduction_count: 0,
        }
    }

    /// Start monitoring (reset treatment timer).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn start_monitoring(&mut self, current_time: f64) {
        self.treatment_start_time = current_time;
        self.violation_detected = false;
        self.warning_count = 0;
        self.reduction_count = 0;
    }

    /// Update metrics and evaluate safety.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn evaluate_safety(
        &mut self,
        metrics: SafetyMetrics,
        current_time: f64,
    ) -> KwaversResult<TherapyAction> {
        self.current_metrics = metrics;

        let action = self.check_limits(current_time);

        if action != TherapyAction::Continue && self.last_action == TherapyAction::Continue {
            match action {
                TherapyAction::Warning => self.warning_count += 1,
                TherapyAction::ReducePower => self.reduction_count += 1,
                TherapyAction::Stop => self.violation_detected = true,
                _ => {}
            }
        }

        self.last_action = action;
        Ok(action)
    }

    /// Update organ dose accumulation.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn accumulate_organ_dose(
        &mut self,
        organ_name: &str,
        dose_increment: f64,
    ) -> KwaversResult<()> {
        if dose_increment < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Organ dose increment must be non-negative".into(),
            ));
        }

        self.organ_doses
            .entry(organ_name.to_string())
            .and_modify(|d| *d += dose_increment)
            .or_insert(dose_increment);

        Ok(())
    }

    /// Get current safety status.
    pub fn status(&self) -> TherapyIntegrationSafetyStatus {
        if self.violation_detected {
            TherapyIntegrationSafetyStatus::TimeLimitExceeded
        } else if self.current_metrics.thermal_index > self.limits.thermal_index_max {
            TherapyIntegrationSafetyStatus::ThermalLimitExceeded
        } else if self.current_metrics.mechanical_index > self.limits.mechanical_index_max {
            TherapyIntegrationSafetyStatus::MechanicalLimitExceeded
        } else if self.current_metrics.cavitation_dose > self.limits.cavitation_dose_max {
            TherapyIntegrationSafetyStatus::CavitationLimitExceeded
        } else {
            TherapyIntegrationSafetyStatus::Safe
        }
    }

    /// Get current metrics.
    pub fn metrics(&self) -> &SafetyMetrics {
        &self.current_metrics
    }

    /// Get event summary.
    pub fn event_summary(&self) -> String {
        format!(
            "Warnings: {}, Power reductions: {}, Violation detected: {}",
            self.warning_count, self.reduction_count, self.violation_detected
        )
    }

    /// Check if therapy should stop.
    pub fn should_stop(&self) -> bool {
        self.violation_detected || self.last_action == TherapyAction::Stop
    }

    /// Get power reduction factor in [0.0, 1.0].
    pub fn power_reduction_factor(&self) -> f64 {
        match self.last_action {
            TherapyAction::Continue => 1.0,
            TherapyAction::Warning => 1.0,
            TherapyAction::ReducePower => 0.5,
            TherapyAction::Stop => 0.0,
        }
    }

    fn check_limits(&self, current_time: f64) -> TherapyAction {
        let elapsed = current_time - self.treatment_start_time;
        // ≥ not > : at exactly max_treatment_time the session must stop.
        if elapsed >= self.limits.max_treatment_time {
            return TherapyAction::Stop;
        }

        let time_margin = self
            .warning_margins
            .get("treatment_time")
            .copied()
            .unwrap_or(0.9);
        if elapsed > self.limits.max_treatment_time * time_margin {
            return TherapyAction::ReducePower;
        }

        if self.current_metrics.thermal_index > self.limits.thermal_index_max {
            return TherapyAction::Stop;
        }

        let ti_margin = self
            .warning_margins
            .get("thermal_index")
            .copied()
            .unwrap_or(0.8);
        if self.current_metrics.thermal_index > self.limits.thermal_index_max * ti_margin {
            return TherapyAction::ReducePower;
        }

        if self.current_metrics.mechanical_index > self.limits.mechanical_index_max {
            return TherapyAction::Stop;
        }

        let mi_margin = self
            .warning_margins
            .get("mechanical_index")
            .copied()
            .unwrap_or(0.8);
        if self.current_metrics.mechanical_index > self.limits.mechanical_index_max * mi_margin {
            return TherapyAction::Warning;
        }

        if self.current_metrics.cavitation_dose > self.limits.cavitation_dose_max {
            return TherapyAction::Stop;
        }

        let cav_margin = self
            .warning_margins
            .get("cavitation_dose")
            .copied()
            .unwrap_or(0.8);
        if self.current_metrics.cavitation_dose > self.limits.cavitation_dose_max * cav_margin {
            return TherapyAction::Warning;
        }

        for (organ, max_dose) in &self.organ_max_doses {
            if let Some(current_dose) = self.organ_doses.get(organ) {
                if current_dose > max_dose {
                    return TherapyAction::Stop;
                }
                if *current_dose > max_dose * 0.8 {
                    return TherapyAction::Warning;
                }
            }
        }

        TherapyAction::Continue
    }
}
