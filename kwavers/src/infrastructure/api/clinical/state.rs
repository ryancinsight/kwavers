use crate::core::error::KwaversResult;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

#[cfg(feature = "pinn")]
use crate::clinical::imaging::workflows::neural::{
    AIBeamformingConfig, AIEnhancedBeamformingProcessor,
};

#[cfg(not(feature = "pinn"))]
use crate::api::{DeviceInfo, DeviceType, DeviceCapability, DeviceStatus};
#[cfg(not(feature = "pinn"))]
use crate::clinical::{ClinicalWorkflow, WorkflowType, WorkflowStatus, WorkflowStep, StepType, StepStatus};

use crate::infrastructure::api::UltrasoundDevice;
use super::dicom::DICOMService;
use super::mobile::MobileOptimizer;

/// Active clinical session tracking
#[derive(Debug, Clone)]
pub struct ClinicalSession {
    /// Session identifier
    pub session_id: String,
    /// Device identifier
    pub device_id: String,
    /// Patient identifier
    pub patient_id: String,
    /// Exam type
    pub exam_type: String,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    /// Session priority
    pub priority: crate::api::AnalysisPriority,
}

/// Clinical API application state
#[derive(Debug, Clone)]
pub struct ClinicalAppState {
    /// AI-enhanced beamforming processor
    #[cfg(feature = "pinn")]
    pub ai_processor: Arc<Mutex<AIEnhancedBeamformingProcessor>>,
    /// Authentication middleware
    pub auth_middleware: Arc<crate::api::auth::AuthMiddleware>,
    /// Connected ultrasound devices registry
    pub device_registry: Arc<RwLock<HashMap<String, UltrasoundDevice>>>,
    /// Active clinical sessions
    pub active_sessions: Arc<RwLock<HashMap<String, ClinicalSession>>>,
    /// DICOM/HL7 integration service
    pub dicom_service: Arc<RwLock<DICOMService>>,
    /// Mobile optimization engine
    pub mobile_optimizer: Arc<RwLock<MobileOptimizer>>,
}

#[cfg(feature = "pinn")]
impl ClinicalAppState {
    /// Create new clinical app state
    pub fn new(auth_middleware: Arc<crate::api::auth::AuthMiddleware>) -> KwaversResult<Self> {
        // Initialize AI processor with default config
        let config = AIBeamformingConfig::default();
        let sensor_positions = vec![
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.001, 0.001, 0.0],
        ];

        let ai_processor = Arc::new(Mutex::new(AIEnhancedBeamformingProcessor::new(
            config,
            sensor_positions,
            None,
        )?));

        Ok(Self {
            ai_processor,
            auth_middleware,
            device_registry: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            dicom_service: Arc::new(RwLock::new(DICOMService::new())),
            mobile_optimizer: Arc::new(RwLock::new(MobileOptimizer::new())),
        })
    }
}

#[cfg(not(feature = "pinn"))]
impl ClinicalAppState {
    /// Create new clinical app state with fallback clinical functionality
    /// Provides basic clinical workflow support without PINN-based ML features
    pub fn new(auth_middleware: Arc<crate::api::auth::AuthMiddleware>) -> KwaversResult<Self> {
        let mut state = Self {
            auth_middleware,
            device_registry: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            dicom_service: Arc::new(RwLock::new(DICOMService::new())),
            mobile_optimizer: Arc::new(RwLock::new(MobileOptimizer::new())),
        };

        // Initialize with basic clinical device configurations
        Self::initialize_basic_devices(&mut state)?;
        Self::initialize_clinical_workflows(&mut state)?;

        Ok(state)
    }

    /// Initialize basic clinical devices for fallback operation
    fn initialize_basic_devices(state: &mut ClinicalAppState) -> KwaversResult<()> {
        let mut registry = state.device_registry.write().unwrap();

        // Register basic ultrasound devices
        registry.insert(
            "default-ultrasound".to_string(),
            DeviceInfo {
                id: "default-ultrasound".to_string(),
                device_type: DeviceType::Ultrasound,
                model: "Basic Ultrasound System".to_string(),
                manufacturer: "KwaverS".to_string(),
                capabilities: vec![
                    DeviceCapability::Imaging2D,
                    DeviceCapability::Doppler,
                    DeviceCapability::BMode,
                ],
                status: DeviceStatus::Available,
                last_calibration: std::time::SystemTime::now(),
                firmware_version: "1.0.0".to_string(),
            },
        );

        // Register basic therapy device
        registry.insert(
            "default-therapy".to_string(),
            DeviceInfo {
                id: "default-therapy".to_string(),
                device_type: DeviceType::Therapy,
                model: "Basic Therapy System".to_string(),
                manufacturer: "KwaverS".to_string(),
                capabilities: vec![DeviceCapability::HIFU, DeviceCapability::Lithotripsy],
                status: DeviceStatus::Available,
                last_calibration: std::time::SystemTime::now(),
                firmware_version: "1.0.0".to_string(),
            },
        );

        Ok(())
    }

    /// Initialize basic clinical workflows
    fn initialize_clinical_workflows(state: &mut ClinicalAppState) -> KwaversResult<()> {
        let mut sessions = state.active_sessions.write().unwrap();

        // Create a default clinical workflow template
        let default_workflow = ClinicalWorkflow {
            id: "default-workflow".to_string(),
            patient_id: "template".to_string(),
            workflow_type: WorkflowType::Diagnostic,
            status: WorkflowStatus::Ready,
            steps: vec![
                WorkflowStep {
                    id: "patient_prep".to_string(),
                    step_type: StepType::Preparation,
                    status: StepStatus::Pending,
                    description: "Patient preparation and positioning".to_string(),
                    estimated_duration: std::time::Duration::from_secs(300), // 5 minutes
                },
                WorkflowStep {
                    id: "imaging".to_string(),
                    step_type: StepType::Imaging,
                    status: StepStatus::Pending,
                    description: "Ultrasound imaging acquisition".to_string(),
                    estimated_duration: std::time::Duration::from_secs(600), // 10 minutes
                },
                WorkflowStep {
                    id: "analysis".to_string(),
                    step_type: StepType::Analysis,
                    status: StepStatus::Pending,
                    description: "Image analysis and interpretation".to_string(),
                    estimated_duration: std::time::Duration::from_secs(300), // 5 minutes
                },
            ],
            created_at: std::time::SystemTime::now(),
            updated_at: std::time::SystemTime::now(),
        };

        sessions.insert("default-workflow".to_string(), default_workflow);
        Ok(())
    }
}
