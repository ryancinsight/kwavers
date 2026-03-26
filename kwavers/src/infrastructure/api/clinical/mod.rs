//! Clinical Ultrasound API Handlers
//!
//! This module provides RESTful endpoints for point-of-care ultrasound integration,
//! including AI-enhanced clinical decision support, device connectivity, and standards compliance.
//!
//! ## Point-of-Care Integration Features
//!
//! - **Real-time AI Analysis**: Clinical decision support with <100ms latency
//! - **Device Connectivity**: Portable ultrasound device integration
//! - **Standards Compliance**: DICOM/HL7 support for clinical workflows
//! - **Mobile Optimization**: Battery-aware processing for portable devices
//! - **Clinical Workflows**: Automated diagnosis and recommendations

pub mod analysis;
pub mod device;
pub mod dicom;
pub mod mobile;
pub mod state;

pub use analysis::{analyze_clinical, get_session_status};
pub use device::{get_device_status, list_devices, register_device};
pub use dicom::{dicom_integrate, DICOMNode, DICOMService};
pub use mobile::{optimize_mobile, MobileOptimizer, OptimizationRule};
pub use state::{ClinicalAppState, ClinicalSession};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::Arc;
    use crate::infrastructure::api::UltrasoundDevice;

    #[tokio::test]
    async fn test_device_registration() {
        // Test complete device registration workflow
        let auth_middleware = Arc::new(
            crate::api::auth::AuthMiddleware::new(
                "test-secret-do-not-use-in-production",
                crate::api::auth::JWTConfig::default(),
            )
            .expect("test auth middleware construction must succeed"),
        );
        let app_state = ClinicalAppState::new(auth_middleware).unwrap();

        // Test device registration via API
        let device_info = UltrasoundDevice {
            device_id: "test-ultrasound-001".to_string(),
            model: "Test Ultrasound System".to_string(),
            capabilities: vec![
                "Imaging2D".to_string(),
                "Doppler".to_string(),
                "ColorFlow".to_string(),
            ],
            imaging_modes: vec!["B-Mode".to_string()],
            max_frame_rate: 30,
            battery_level: None,
            status: crate::api::DeviceStatus::Available,
            last_seen: Utc::now(),
        };

        // Register device
        {
            let mut registry = app_state.device_registry.write().await;
            registry.insert(device_info.device_id.clone(), device_info.clone());
        }

        // Verify device was registered correctly
        {
            let registry = app_state.device_registry.read().await;
            let registered_device = registry.get(&device_info.device_id).unwrap();
            assert_eq!(registered_device.device_id, device_info.device_id);
            assert_eq!(registered_device.capabilities.len(), 3);
            assert_eq!(
                registered_device.status,
                crate::api::DeviceStatus::Available
            );
        }

        // Test device status update
        {
            let mut registry = app_state.device_registry.write().await;
            if let Some(device) = registry.get_mut(&device_info.device_id) {
                device.status = crate::api::DeviceStatus::InUse;
            }
        }

        // Verify status update
        {
            let registry = app_state.device_registry.read().await;
            let updated_device = registry.get(&device_info.device_id).unwrap();
            assert_eq!(updated_device.status, crate::api::DeviceStatus::InUse);
        }
    }

    #[test]
    fn test_clinical_analysis_structure() {
        // Test clinical analysis data structures
        let finding = crate::api::ClinicalFinding {
            finding_type: crate::api::FindingType::Lesion,
            description: "Test lesion".to_string(),
            confidence: 0.85,
            location: [10.0, 20.0, 30.0],
            measurements: crate::api::FindingMeasurements {
                max_diameter_mm: 5.0,
                area_volume: 78.5,
                boundary_confidence: 0.8,
            },
            clinical_significance: 0.7,
        };

        assert_eq!(finding.confidence, 0.85);
        assert!(matches!(
            finding.finding_type,
            crate::api::FindingType::Lesion
        ));
    }

    #[test]
    fn test_dicom_caching_behavior() {
        // Setup temp dir and files
        let temp_dir =
            std::env::temp_dir().join(format!("kwavers_dicom_test_{}", uuid::Uuid::new_v4()));
        let study_uid = "study1";
        let study_path = temp_dir.join(study_uid);
        std::fs::create_dir_all(&study_path).unwrap();
        // Create dummy files so read_directory finds something (though it returns empty study)
        std::fs::write(study_path.join("file1.dcm"), b"dummy").unwrap();

        // Setup service
        let mut service = DICOMService::new();
        service.dicom_nodes.insert(
            "local".to_string(),
            DICOMNode {
                node_id: "local".to_string(),
                ae_title: "LOCAL".to_string(),
                host: "localhost".to_string(),
                port: 104,
                last_seen: Utc::now(),
                storage_directory: Some(temp_dir.to_string_lossy().to_string()),
            },
        );

        // First read
        let study1 = service.read_study(study_uid).unwrap().unwrap();

        // Second read
        let study2 = service.read_study(study_uid).unwrap().unwrap();

        // Check if they are same Arc
        assert!(
            std::sync::Arc::ptr_eq(&study1, &study2),
            "Cache should return the same Arc"
        );

        // Cleanup
        std::fs::remove_dir_all(temp_dir).unwrap();
    }
}
