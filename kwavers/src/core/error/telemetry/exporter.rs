use tracing::{debug, info};

use crate::core::error::{KwaversError, RecoveryAttempt};

use super::correlation::TelemetryContext;

/// Telemetry exporter for error metrics and recovery attempts.
pub trait TelemetryExporter: std::fmt::Debug + Send + Sync {
    /// Export an error event.
    fn export_error(&self, error: &KwaversError, context: &TelemetryContext);

    /// Export a recovery attempt.
    fn export_recovery(&self, attempt: &RecoveryAttempt);

    /// Flush any buffered state.
    fn flush(&self);

    /// Shutdown exporter gracefully.
    fn shutdown(&self);
}

/// Console exporter for development diagnostics.
#[derive(Debug)]
pub struct ConsoleExporter;

impl TelemetryExporter for ConsoleExporter {
    fn export_error(&self, error: &KwaversError, context: &TelemetryContext) {
        debug!(
            error = %error,
            trace_id = %context.trace_id,
            service = %context.service_name,
            "Console exporter: error event"
        );
    }

    fn export_recovery(&self, attempt: &RecoveryAttempt) {
        debug!(
            strategy = %attempt.strategy,
            succeeded = attempt.succeeded,
            "Console exporter: recovery event"
        );
    }

    fn flush(&self) {}

    fn shutdown(&self) {
        info!("Console exporter shutdown");
    }
}
