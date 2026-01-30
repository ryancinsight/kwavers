//! Real-Time Safety Control System for Therapy Execution
//!
//! This module implements active safety enforcement for therapeutic ultrasound,
//! enabling real-time monitoring and adaptive therapy control to prevent adverse effects.
//!
//! # Design Pattern
//!
//! The SafetyController follows an event-driven architecture:
//! 1. Monitor current safety metrics against configured limits
//! 2. Detect warning conditions (80% of limit)
//! 3. Detect violation conditions (exceeds limit)
//! 4. Generate appropriate control actions
//! 5. Report events to clinical UI
//!
//! # Enforced Safety Limits
//!
//! - **Thermal Index (TI)**: IEC 62359 formula, max 6.0 for soft tissue therapy
//! - **Mechanical Index (MI)**: FDA formula, max 1.9 for most applications
//! - **Cavitation Dose**: Integrated cavitation activity, max 1.0
//! - **Treatment Time**: ALARA principle, max ~600 seconds per session
//! - **Organ Dose**: Patient-specific risk organ constraints
//!
//! # Safety Action Hierarchy
//!
//! ```text
//! Safe ──→ Warning ──→ Reduced Power ──→ Stop
//! (< 80%)   (80-100%)   (reduced by 50%)   (> limit)
//! ```
//!
//! # Clinical Compliance
//!
//! - IEC 60601-2-49 (therapeutic ultrasound equipment)
//! - FDA 510(k) Guidance for ultrasound safety
//! - AIUM NEMA standards for output measurement

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

use super::orchestrator::safety::{SafetyLimits, SafetyMetrics, SafetyStatus};

/// Action to take based on safety status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyAction {
    /// Continue therapy normally
    Continue,
    /// Warning: approaching limit, recommend monitoring
    Warning,
    /// Reduce acoustic power to 50% of current
    ReducePower,
    /// Immediately stop therapy (limit exceeded)
    Stop,
}

impl TherapyAction {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            TherapyAction::Continue => "Therapy safe to continue",
            TherapyAction::Warning => "Approaching safety limit - monitoring recommended",
            TherapyAction::ReducePower => "Safety margin exceeded - reducing power",
            TherapyAction::Stop => "Safety limit exceeded - stopping therapy",
        }
    }

    /// Get priority (higher = more urgent)
    pub fn priority(&self) -> u8 {
        match self {
            TherapyAction::Continue => 0,
            TherapyAction::Warning => 1,
            TherapyAction::ReducePower => 2,
            TherapyAction::Stop => 3,
        }
    }
}

/// Real-time safety controller for therapeutic ultrasound
///
/// Monitors safety metrics and enforces limits during therapy execution.
/// Generates control actions to prevent adverse tissue effects.
#[derive(Debug, Clone)]
pub struct SafetyController {
    /// Configured safety limits
    limits: SafetyLimits,

    /// Warning margins (typically 80% of limit)
    /// Values in range [0.0, 1.0] representing fraction of limit
    warning_margins: HashMap<String, f64>,

    /// Current safety metrics
    current_metrics: SafetyMetrics,

    /// Per-organ accumulated dose (in arbitrary units)
    organ_doses: HashMap<String, f64>,

    /// Maximum allowed dose per organ
    organ_max_doses: HashMap<String, f64>,

    /// Treatment start time (seconds)
    treatment_start_time: f64,

    /// Whether a violation has occurred (sticky flag)
    violation_detected: bool,

    /// Last action taken
    last_action: TherapyAction,

    /// Number of warning events
    warning_count: u32,

    /// Number of power reduction events
    reduction_count: u32,
}

impl SafetyController {
    /// Create new safety controller
    ///
    /// # Arguments
    ///
    /// - `limits`: Safety limits configuration
    /// - `organ_limits`: Per-organ maximum doses (optional)
    ///
    /// # Returns
    ///
    /// New controller with metrics initialized to zero
    pub fn new(limits: SafetyLimits, organ_limits: Option<HashMap<String, f64>>) -> Self {
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

    /// Start monitoring (reset treatment timer)
    pub fn start_monitoring(&mut self, current_time: f64) {
        self.treatment_start_time = current_time;
        self.violation_detected = false;
        self.warning_count = 0;
        self.reduction_count = 0;
    }

    /// Update metrics and evaluate safety
    ///
    /// # Arguments
    ///
    /// - `metrics`: Current safety metrics from therapy execution
    /// - `current_time`: Current simulation time (seconds)
    ///
    /// # Returns
    ///
    /// Action to take based on safety status
    pub fn evaluate_safety(
        &mut self,
        metrics: SafetyMetrics,
        current_time: f64,
    ) -> KwaversResult<TherapyAction> {
        // Update metrics
        self.current_metrics = metrics;

        // Check each limit in priority order
        let action = self.check_limits(current_time);

        // Update action history
        if action != TherapyAction::Continue && self.last_action == TherapyAction::Continue {
            // First time transitioning from safe
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

    /// Check all safety limits and return appropriate action
    fn check_limits(&self, current_time: f64) -> TherapyAction {
        // Check treatment time (highest priority for ALARA)
        let elapsed = current_time - self.treatment_start_time;
        if elapsed > self.limits.max_treatment_time {
            return TherapyAction::Stop;
        }

        // Check treatment time warning
        let time_margin = self
            .warning_margins
            .get("treatment_time")
            .copied()
            .unwrap_or(0.9);
        if elapsed > self.limits.max_treatment_time * time_margin {
            return TherapyAction::ReducePower;
        }

        // Check thermal index (most dangerous)
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

        // Check mechanical index
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

        // Check cavitation dose
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

        // Check organ doses
        for (organ, max_dose) in &self.organ_max_doses {
            if let Some(current_dose) = self.organ_doses.get(organ) {
                if current_dose > max_dose {
                    return TherapyAction::Stop;
                }

                // Warning at 80% of limit
                if current_dose > max_dose * 0.8 {
                    return TherapyAction::Warning;
                }
            }
        }

        TherapyAction::Continue
    }

    /// Update organ dose accumulation
    ///
    /// # Arguments
    ///
    /// - `organ_name`: Name of organ
    /// - `dose_increment`: Incremental dose to add (should be dt-normalized)
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

    /// Get current safety status
    pub fn status(&self) -> SafetyStatus {
        if self.violation_detected {
            SafetyStatus::TimeLimitExceeded
        } else if self.current_metrics.thermal_index > self.limits.thermal_index_max {
            SafetyStatus::ThermalLimitExceeded
        } else if self.current_metrics.mechanical_index > self.limits.mechanical_index_max {
            SafetyStatus::MechanicalLimitExceeded
        } else if self.current_metrics.cavitation_dose > self.limits.cavitation_dose_max {
            SafetyStatus::CavitationLimitExceeded
        } else {
            SafetyStatus::Safe
        }
    }

    /// Get current metrics
    pub fn metrics(&self) -> &SafetyMetrics {
        &self.current_metrics
    }

    /// Get event summary
    pub fn event_summary(&self) -> String {
        format!(
            "Warnings: {}, Power reductions: {}, Violation detected: {}",
            self.warning_count, self.reduction_count, self.violation_detected
        )
    }

    /// Check if therapy should stop
    pub fn should_stop(&self) -> bool {
        self.violation_detected || self.last_action == TherapyAction::Stop
    }

    /// Get power reduction factor (for adaptive control)
    ///
    /// Returns value in [0.0, 1.0] representing fraction of nominal power
    /// - 1.0: nominal power
    /// - 0.5: 50% power (during ReducePower action)
    /// - 0.0: zero power (stop)
    pub fn power_reduction_factor(&self) -> f64 {
        match self.last_action {
            TherapyAction::Continue => 1.0,
            TherapyAction::Warning => 1.0,
            TherapyAction::ReducePower => 0.5,
            TherapyAction::Stop => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_controller() -> SafetyController {
        let limits = SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1.0,
            max_treatment_time: 600.0,
        };

        SafetyController::new(limits, None)
    }

    #[test]
    fn test_controller_creation() {
        let controller = create_test_controller();
        assert_eq!(controller.last_action, TherapyAction::Continue);
        assert!(!controller.violation_detected);
    }

    #[test]
    fn test_thermal_index_violation() {
        let mut controller = create_test_controller();
        controller.start_monitoring(0.0);

        let mut metrics = SafetyMetrics::default();
        metrics.thermal_index = 6.5; // Exceeds limit of 6.0

        let action = controller.evaluate_safety(metrics, 1.0).unwrap();
        assert_eq!(action, TherapyAction::Stop);
    }

    #[test]
    fn test_thermal_index_warning() {
        let mut controller = create_test_controller();
        controller.start_monitoring(0.0);

        let mut metrics = SafetyMetrics::default();
        metrics.thermal_index = 5.0; // 83% of 6.0 limit

        let action = controller.evaluate_safety(metrics, 1.0).unwrap();
        assert_eq!(action, TherapyAction::ReducePower);
    }

    #[test]
    fn test_mechanical_index_safe() {
        let mut controller = create_test_controller();
        controller.start_monitoring(0.0);

        let mut metrics = SafetyMetrics::default();
        metrics.mechanical_index = 1.5; // Below 1.9 limit

        let action = controller.evaluate_safety(metrics, 1.0).unwrap();
        assert_eq!(action, TherapyAction::Continue);
    }

    #[test]
    fn test_cavitation_dose_exceeds() {
        let mut controller = create_test_controller();
        controller.start_monitoring(0.0);

        let mut metrics = SafetyMetrics::default();
        metrics.cavitation_dose = 1.1; // Exceeds 1.0 limit

        let action = controller.evaluate_safety(metrics, 1.0).unwrap();
        assert_eq!(action, TherapyAction::Stop);
    }

    #[test]
    fn test_treatment_time_limit() {
        let mut controller = create_test_controller();
        controller.start_monitoring(0.0);

        let metrics = SafetyMetrics::default();

        // At max time
        let action = controller.evaluate_safety(metrics.clone(), 600.0).unwrap();
        assert_eq!(action, TherapyAction::Stop);
    }

    #[test]
    fn test_power_reduction_factor() {
        let mut controller = create_test_controller();

        assert_eq!(controller.power_reduction_factor(), 1.0); // Continue

        controller.last_action = TherapyAction::ReducePower;
        assert_eq!(controller.power_reduction_factor(), 0.5);

        controller.last_action = TherapyAction::Stop;
        assert_eq!(controller.power_reduction_factor(), 0.0);
    }

    #[test]
    fn test_organ_dose_tracking() {
        let limits = SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1.0,
            max_treatment_time: 600.0,
        };

        let mut organ_limits = HashMap::new();
        organ_limits.insert("brain".to_string(), 100.0);

        let mut controller = SafetyController::new(limits, Some(organ_limits));
        controller.start_monitoring(0.0);

        // Accumulate dose
        controller.accumulate_organ_dose("brain", 50.0).unwrap();
        controller.accumulate_organ_dose("brain", 40.0).unwrap();

        // Should trigger warning at 80%
        let mut metrics = SafetyMetrics::default();
        let action = controller.evaluate_safety(metrics, 1.0).unwrap();
        assert_eq!(action, TherapyAction::Warning);

        // Further accumulation should trigger stop
        controller.accumulate_organ_dose("brain", 15.0).unwrap();
        let action = controller.evaluate_safety(metrics, 1.0).unwrap();
        assert_eq!(action, TherapyAction::Stop);
    }

    #[test]
    fn test_event_summary() {
        let mut controller = create_test_controller();
        controller.start_monitoring(0.0);

        let metrics = SafetyMetrics::default();
        controller.evaluate_safety(metrics, 1.0).unwrap();

        let summary = controller.event_summary();
        assert!(summary.contains("Warnings:"));
        assert!(summary.contains("Power reductions:"));
    }
}
