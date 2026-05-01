//! Value-semantic regression tests for API model serialization and defaults.

use super::*;
use chrono::Utc;
use std::collections::HashMap;

use crate::infrastructure::api::TrainingConfig;

#[test]
fn test_job_queue_entry_defaults() {
    let entry = JobQueueEntry::default();
    assert_eq!(entry.priority, 0);
    assert!(matches!(entry.status, crate::api::JobStatus::Queued));
    assert!(entry.progress.is_none());
}

#[test]
fn test_training_config_defaults() {
    let config = TrainingConfig::default();
    assert_eq!(config.collocation_points, 1000);
    assert_eq!(config.batch_size, 32);
    assert_eq!(config.epochs, 100);
}

#[test]
fn test_audit_log_serialization() {
    let entry = AuditLogEntry {
        id: "audit_123".to_string(),
        timestamp: Utc::now(),
        user_id: Some("user_456".to_string()),
        action: "train_pinn".to_string(),
        resource: "job".to_string(),
        resource_id: Some("job_789".to_string()),
        ip_address: "192.168.1.1".to_string(),
        user_agent: "PINN-API-Client/1.0".to_string(),
        success: true,
        metadata: Some(HashMap::from([(
            "duration_ms".to_string(),
            serde_json::json!(1500),
        )])),
    };

    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: AuditLogEntry = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.action, "train_pinn");
    assert!(deserialized.success);
}
