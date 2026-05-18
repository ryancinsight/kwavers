//! Error telemetry and observability integration.
//!
//! This module keeps `tracing` as the in-process correlation primitive and
//! layers exporter-optional metrics/alerting on top.

mod alerting;
mod correlation;
mod exporter;
mod ids;
mod metrics;
mod severity;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{error, info, warn};

use crate::core::error::{KwaversError, RecoveryAttempt};

pub use alerting::AlertThreshold;
use alerting::BreachTracker;
pub use correlation::TelemetryContext;
pub use exporter::{ConsoleExporter, TelemetryExporter};
pub use metrics::{TelemetryErrorCounts, MetricType};
pub use severity::ErrorSeverity;

/// Central telemetry hub for error observability.
#[derive(Debug)]
pub struct ErrorTelemetry {
    metrics: Arc<TelemetryErrorCounts>,
    exporters: Vec<Box<dyn TelemetryExporter>>,
    alert_thresholds: HashMap<ErrorSeverity, AlertThreshold>,
    breach_tracker: BreachTracker,
}

impl ErrorTelemetry {
    /// Create a telemetry hub with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(TelemetryErrorCounts::new()),
            exporters: Vec::new(),
            alert_thresholds: Self::default_thresholds(),
            breach_tracker: BreachTracker::default(),
        }
    }

    fn default_thresholds() -> HashMap<ErrorSeverity, AlertThreshold> {
        HashMap::from([
            (
                ErrorSeverity::Critical,
                AlertThreshold {
                    severity: ErrorSeverity::Critical,
                    max_errors_per_minute: ErrorSeverity::Critical.threshold(),
                    consecutive_breaches: 1,
                    window_seconds: 60,
                },
            ),
            (
                ErrorSeverity::High,
                AlertThreshold {
                    severity: ErrorSeverity::High,
                    max_errors_per_minute: ErrorSeverity::High.threshold(),
                    consecutive_breaches: 3,
                    window_seconds: 60,
                },
            ),
            (
                ErrorSeverity::Medium,
                AlertThreshold {
                    severity: ErrorSeverity::Medium,
                    max_errors_per_minute: ErrorSeverity::Medium.threshold(),
                    consecutive_breaches: 3,
                    window_seconds: 60,
                },
            ),
            (
                ErrorSeverity::Low,
                AlertThreshold {
                    severity: ErrorSeverity::Low,
                    max_errors_per_minute: ErrorSeverity::Low.threshold(),
                    consecutive_breaches: 3,
                    window_seconds: 60,
                },
            ),
        ])
    }

    /// Register an exporter.
    #[must_use]
    pub fn with_exporter<E: TelemetryExporter + 'static>(mut self, exporter: E) -> Self {
        self.exporters.push(Box::new(exporter));
        self
    }

    /// Record an error and update metrics/alerting.
    pub fn record_error(&self, error: &KwaversError, context: &TelemetryContext) {
        self.metrics.record_error(error);

        for exporter in &self.exporters {
            exporter.export_error(error, context);
        }

        self.check_thresholds(error);

        info!(
            trace_id = %context.trace_id,
            severity = ?ErrorSeverity::from(error),
            "Error recorded in telemetry"
        );
    }

    /// Record a recovery attempt.
    pub fn record_recovery(&self, attempt: &RecoveryAttempt, _context: &TelemetryContext) {
        self.metrics.record_recovery_attempt();
        if attempt.succeeded {
            self.metrics.record_recovery_success();
            info!(
                strategy = %attempt.strategy,
                duration_ms = attempt.duration.as_millis(),
                "Recovery succeeded"
            );
        } else {
            warn!(
                strategy = %attempt.strategy,
                error = %attempt.original_error,
                "Recovery failed"
            );
        }

        for exporter in &self.exporters {
            exporter.export_recovery(attempt);
        }
    }

    fn check_thresholds(&self, error: &KwaversError) {
        let severity = ErrorSeverity::from(error);
        let Some(threshold) = self.alert_thresholds.get(&severity) else {
            return;
        };

        let current_rate = self
            .metrics
            .error_rate_per_minute(severity, threshold.window_seconds);
        let streak = self
            .breach_tracker
            .update(severity, current_rate > threshold.max_errors_per_minute);

        if streak >= threshold.consecutive_breaches {
            self.metrics.record_threshold_breach(severity);
            error!(
                severity = ?severity,
                current_rate,
                threshold = threshold.max_errors_per_minute,
                streak,
                "ALERT: error rate threshold exceeded"
            );
        }
    }

    #[must_use]
    pub fn breach_streak(&self, severity: ErrorSeverity) -> u32 {
        self.breach_tracker.streak(severity)
    }

    #[must_use]
    pub fn metrics(&self) -> &TelemetryErrorCounts {
        &self.metrics
    }

    #[must_use]
    pub fn prometheus_metrics(&self) -> String {
        self.metrics.export_prometheus()
    }

    pub fn shutdown(&self) {
        for exporter in &self.exporters {
            exporter.shutdown();
        }
        info!("Error telemetry shutdown complete");
    }
}

impl Default for ErrorTelemetry {
    fn default() -> Self {
        Self::new()
    }
}
