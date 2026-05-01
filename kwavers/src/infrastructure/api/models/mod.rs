//! Additional API data models
//!
//! This module contains supplementary data models used by the PINN API
//! for job queuing, result storage, and operational metadata.
//!
//! ## Clinical Ultrasound API
//!
//! Additional models for point-of-care ultrasound integration:
//! - Device connectivity and real-time imaging
//! - AI-enhanced clinical decision support
//! - DICOM/HL7 standards compliance
//! - Mobile-optimized workflows
//!
//! ## Module layout
//!
//! - [`jobs`]: PINN training job queue, training results, validation,
//!   benchmarks, API usage, system health, audit log, notifications.
//! - [`devices`]: ultrasound device connectivity, capability, status.
//! - [`imaging`]: real-time `UltrasoundFrame` and `ImagingParameters`.
//! - [`clinical`]: clinical analysis request/response, findings, tissue
//!   characterization, recommendations, processing/quality metrics.
//! - [`dicom`]: DICOM standards-compliance integration types.
//! - [`mobile`]: mobile-optimized workflow and constraint types.

mod clinical;
mod devices;
mod dicom;
mod imaging;
mod jobs;
mod mobile;

#[cfg(test)]
mod tests;

pub use clinical::{
    AbnormalRegion, AnalysisPriority, ClinicalAnalysisRequest, ClinicalAnalysisResponse,
    ClinicalContext, ClinicalFinding, ClinicalRecommendation, FindingMeasurements, FindingType,
    OperatorLevel, ProcessingMetrics, QualityIndicators, RecommendationType, TissueCharacterization,
    TissueProperties, TissueRegion, UrgencyLevel,
};
pub use devices::{
    DeviceCapability, DeviceInfo, DeviceStatus, DeviceType, UltrasoundDevice,
};
pub use dicom::{DICOMIntegrationRequest, DICOMIntegrationResponse, DICOMStudyInfo, DICOMValue};
pub use imaging::{ImagingParameters, UltrasoundFrame};
pub use jobs::{
    APIUsageStats, AuditLogEntry, DeliveryMethod, JobProgress, JobQueueEntry, NotificationConfig,
    NotificationType, PINNJobConfig, PerformanceBenchmarks, SystemHealthMetrics, TrainingResult,
    ValidationResults,
};
pub use mobile::{
    ConnectionType, DeviceCapabilities, MobileOptimizationRequest, MobileOptimizationResponse,
    NetworkConditions, PerformancePredictions, PerformanceTargets, PowerEstimates, PowerSettings,
    ProcessingConfig,
};
