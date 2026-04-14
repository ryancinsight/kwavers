use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::core::error::context::TelemetrySpan;

use super::ids::instance_id;

/// Session-level telemetry metadata wrapping a captured per-error span snapshot.
#[derive(Debug, Clone)]
pub struct TelemetryContext {
    /// Trace ID from tracing / distributed correlation.
    pub trace_id: String,
    /// Span ID from tracing / distributed correlation.
    pub span_id: String,
    /// Parent span if available.
    pub parent_span_id: Option<String>,
    /// Wall-clock timestamp when the context was created.
    pub timestamp: SystemTime,
    /// Logical service name.
    pub service_name: &'static str,
    /// Service version.
    pub service_version: String,
    /// Host or process identity.
    pub instance_id: String,
    /// Additional exporter attributes.
    pub attributes: HashMap<String, String>,
}

impl TelemetryContext {
    /// Create a telemetry context by capturing the active tracing span.
    #[must_use]
    pub fn from_current_span() -> Self {
        let span = TelemetrySpan::capture();
        let timestamp = UNIX_EPOCH + Duration::from_millis(span.timestamp_ms);

        Self {
            trace_id: span.trace_id,
            span_id: span.span_id,
            parent_span_id: span.parent_span_id,
            timestamp,
            service_name: span.service_name,
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            instance_id: instance_id(),
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute for exporters.
    #[must_use]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Timestamp in milliseconds since Unix epoch.
    #[must_use]
    pub fn timestamp_ms(&self) -> u128 {
        self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }

    /// Convert to the lightweight per-error span snapshot.
    #[must_use]
    pub fn as_span(&self) -> TelemetrySpan {
        TelemetrySpan {
            trace_id: self.trace_id.clone(),
            span_id: self.span_id.clone(),
            parent_span_id: self.parent_span_id.clone(),
            timestamp_ms: self.timestamp_ms() as u64,
            service_name: self.service_name,
        }
    }
}
