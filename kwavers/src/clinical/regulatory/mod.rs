pub mod device;
pub mod evidence;
pub mod performance;
pub mod risk;
pub mod submission;

pub use device::{DeviceClass, DeviceDescription, PredicateDevice};
pub use evidence::ClinicalEvidence;
pub use performance::PerformanceTest;
pub use risk::RiskRecord;
pub use submission::SubmissionDocument;

// ============================================================================
// Helper Functions
// ============================================================================

use std::time::{SystemTime, UNIX_EPOCH};

/// Get current ISO 8601 datetime string
pub(crate) fn iso8601_now() -> String {
    let now = SystemTime::now();
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();

    let seconds = duration.as_secs();
    let days_since_epoch = seconds / 86400;
    let year = 1970 + days_since_epoch / 365;
    let month = (days_since_epoch % 365) / 30 + 1;
    let day = (days_since_epoch % 365) % 30 + 1;

    format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_class() {
        assert_eq!(DeviceClass::ClassII.as_str(), "Class II");
        assert_eq!(DeviceClass::ClassII.risk_level(), "Moderate Risk");
    }

    #[test]
    fn test_device_description_creation() {
        let desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        assert_eq!(desc.name, "HIFU Device");
        assert_eq!(desc.classification, DeviceClass::ClassII);
    }

    #[test]
    fn test_device_description_validation() {
        let desc = DeviceDescription::new("", DeviceClass::ClassII);
        assert!(desc.validate().is_err());

        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        assert!(desc.validate().is_err()); // No intended use

        desc.intended_use = "Ablation therapy".to_string();
        assert!(desc.validate().is_err()); // No indications

        desc.add_indication("Benign tumors");
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn test_predicate_device() {
        let predicate = PredicateDevice::new("Device A", "K060123", "Manufacturer");
        assert_eq!(predicate.k_number, "K060123");
    }

    #[test]
    fn test_risk_record_rpn_calculation() {
        let mut risk = RiskRecord::new("R001", "Device failure");
        risk.severity = "High".to_string();
        risk.probability = "Medium".to_string();

        let rpn = risk.calculate_rpn();
        assert_eq!(rpn, 7 * 3); // 21
    }

    #[test]
    fn test_performance_test() {
        let test =
            PerformanceTest::new("TEST001", "Safety Test").with_objective("Verify device safety");
        assert_eq!(test.objective, "Verify device safety");
        assert!(test.result.is_none());
    }

    #[test]
    fn test_clinical_evidence() {
        let mut evidence = ClinicalEvidence::new("EV001", "Clinical Study");
        evidence.subject_count = 50;
        evidence.primary_outcome = "Pain reduction".to_string();

        assert!(evidence.validate().is_ok());
    }

    #[test]
    fn test_submission_document_creation() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let doc = SubmissionDocument::new("Manufacturer Inc", desc);
        assert!(doc.is_ok());

        let doc = doc.unwrap();
        assert_eq!(doc.status, "Draft");
    }

    #[test]
    fn test_submission_checklist() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let doc = SubmissionDocument::new("Manufacturer Inc", desc).unwrap();
        let checklist = doc.generate_checklist();

        assert!(checklist.contains_key("Device Description Complete"));
        assert!(*checklist.get("Device Description Complete").unwrap());
    }

    #[test]
    fn test_submission_validation() {
        let mut desc = DeviceDescription::new("HIFU Device", DeviceClass::ClassII);
        desc.intended_use = "Ablation therapy".to_string();
        desc.add_indication("Benign tumors");

        let mut doc = SubmissionDocument::new("Manufacturer Inc", desc).unwrap();

        // Should fail - no predicate device
        assert!(doc.validate().is_err());

        // Add predicate
        let predicate = PredicateDevice::new("Device A", "K060123", "Manufacturer");
        let mut predicate_fixed = predicate.clone();
        predicate_fixed.add_similarity("Same intended use");
        doc.add_predicate(predicate_fixed).ok();

        // Should fail - no performance tests
        assert!(doc.validate().is_err());

        // Add test
        let mut test = PerformanceTest::new("TEST001", "Safety Test");
        test.objective = "Verify safety".to_string();
        test.method = "ISO standard".to_string();
        test.record_result(true);
        doc.add_performance_test(test).ok();

        // Should fail - no summary
        assert!(doc.validate().is_err());

        doc.summary = "This device is equivalent to predicate".to_string();

        // Should pass now
        assert!(doc.validate().is_ok());
    }
}
