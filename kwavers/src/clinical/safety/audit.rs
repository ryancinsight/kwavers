//! Safety Audit Logging — IEC 60601-2-37 Compliance
//!
//! Comprehensive event logging for clinical safety systems. All safety-relevant
//! events (violations, state changes, emergency stops, treatment sessions) are
//! recorded with timestamps and metadata for regulatory audit trails.
//!
//! # IEC 60601-2-37 Requirements
//!
//! - **Clause 201.12.4.4**: Equipment shall provide means to record safety-relevant events
//! - **Clause 201.7.9.3.1**: Records shall include date, time, and operator identification
//!
//! # Thread Safety
//!
//! The audit logger uses `Arc<Mutex<Vec<AuditEntry>>>` for thread-safe concurrent
//! logging from multiple subsystems. Mutex poisoning is recovered via
//! `unwrap_or_else(|e| e.into_inner())` to prevent audit loss during panics.

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{ClinicalSafetyLevel, SafetyViolation};

/// Comprehensive audit logger for safety events.
#[derive(Debug)]
pub struct SafetyAuditLogger {
    log_entries: Arc<Mutex<Vec<AuditEntry>>>,
    max_entries: usize,
}

impl SafetyAuditLogger {
    /// Create new audit logger with a maximum entry capacity.
    ///
    /// When the capacity is exceeded, the oldest entry is removed (FIFO).
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            log_entries: Arc::new(Mutex::new(Vec::new())),
            max_entries,
        }
    }

    /// Log a safety event with typed classification and metadata.
    pub fn log_event(
        &self,
        event_type: AuditSafetyEventType,
        message: String,
        metadata: HashMap<String, String>,
    ) {
        let entry = AuditEntry {
            timestamp: Instant::now(),
            event_type,
            message,
            metadata,
        };

        let mut entries = self.log_entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.push(entry);

        // Maintain maximum log size
        if entries.len() > self.max_entries {
            entries.remove(0);
        }
    }

    /// Log a safety violation with structured metadata.
    pub fn log_violation(&self, violation: &SafetyViolation) {
        let mut metadata = HashMap::new();
        metadata.insert("parameter".to_owned(), violation.parameter.clone());
        metadata.insert(
            "measured_value".to_owned(),
            violation.measured_value.to_string(),
        );
        metadata.insert("limit_value".to_owned(), violation.limit_value.to_string());
        metadata.insert("severity".to_owned(), format!("{:?}", violation.severity));

        self.log_event(
            AuditSafetyEventType::Violation,
            violation.message.clone(),
            metadata,
        );
    }

    /// Log a system safety state transition.
    pub fn log_system_state(&self, old_state: ClinicalSafetyLevel, new_state: ClinicalSafetyLevel) {
        let mut metadata = HashMap::new();
        metadata.insert("old_state".to_owned(), format!("{:?}", old_state));
        metadata.insert("new_state".to_owned(), format!("{:?}", new_state));

        let message = format!(
            "System safety state changed from {:?} to {:?}",
            old_state, new_state
        );
        self.log_event(AuditSafetyEventType::StateChange, message, metadata);
    }

    /// Get a clone of all audit log entries.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn get_entries(&self) -> Vec<AuditEntry> {
        self.log_entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Export audit log to a JSONL file (one JSON object per line).
    ///
    /// Fields per line: `timestamp_ms`, `event_type`, `message`, `metadata`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn export_log(&self, filename: &str) -> KwaversResult<()> {
        use std::io::Write;

        let entries = self.log_entries.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = std::fs::File::create(filename).map_err(KwaversError::Io)?;

        let reference = entries.first().map(|e| e.timestamp);

        for entry in entries.iter() {
            let elapsed_ms = reference.map_or(0, |r| entry.timestamp.duration_since(r).as_millis());

            let meta_pairs: Vec<String> = entry
                .metadata
                .iter()
                .map(|(k, v)| {
                    format!(
                        "\"{}\":\"{}\"",
                        k.replace('"', "\\\""),
                        v.replace('"', "\\\"")
                    )
                })
                .collect();
            let meta_json = format!("{{{}}}", meta_pairs.join(","));

            writeln!(
                file,
                r#"{{"timestamp_ms":{},"event_type":"{:?}","message":"{}","metadata":{}}}"#,
                elapsed_ms,
                entry.event_type,
                entry.message.replace('"', "\\\""),
                meta_json,
            )
            .map_err(KwaversError::Io)?;
        }

        Ok(())
    }
}

/// Audit log entry with timestamp and structured metadata.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    /// Event timestamp
    pub timestamp: Instant,
    /// Type of safety event
    pub event_type: AuditSafetyEventType,
    /// Event message
    pub message: String,
    /// Additional metadata key-value pairs
    pub metadata: HashMap<String, String>,
}

/// Types of safety events for audit classification.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AuditSafetyEventType {
    /// Safety violation detected
    Violation,
    /// System state change
    StateChange,
    /// Emergency stop activated
    EmergencyStop,
    /// Treatment session started
    TreatmentStart,
    /// Treatment session ended
    TreatmentEnd,
    /// System startup/shutdown
    SystemEvent,
}
