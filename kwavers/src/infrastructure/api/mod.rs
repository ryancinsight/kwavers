//! Enterprise API for PINN Services.
//!
//! Provides RESTful APIs for Physics-Informed Neural Network training and inference
//! with JWT authentication, rate limiting, and OpenAPI 3.0 compliance.

pub mod api_types;
pub mod auth;
#[cfg(feature = "pinn")]
pub mod clinical;
pub mod handlers;
pub mod job_manager;
pub mod middleware;
pub mod model_registry;
pub mod models;
pub mod rate_limiter;
pub mod router;
#[cfg(test)]
mod tests;

pub use api_types::{
    APIConfig, APIError, APIErrorType, APIVersion, BoundaryConditionSpec, GeometrySpec,
    HealthCheck, HealthStatus, JobInfoResponse, JobStatus, ListModelsResponse, ModelMetadata,
    ObstacleSpec, PINNInferenceRequest, PINNInferenceResponse, PINNTrainingRequest,
    PINNTrainingResponse, PaginationParams, PhysicsParameters, RateLimitConfig, RateLimitInfo,
    ServiceStatus, TrainingConfig, TrainingMetrics, TrainingProgress,
};

// Clinical ultrasound API type re-exports.
#[cfg(feature = "pinn")]
pub use models::{
    AbnormalRegion, AnalysisPriority, ClinicalAnalysisRequest, ClinicalAnalysisResponse,
    ClinicalContext, ClinicalFinding, ClinicalRecommendation, ConnectionType,
    DICOMIntegrationRequest, DICOMIntegrationResponse, DICOMStudyInfo, DICOMValue,
    DeviceCapabilities, DeviceCapability, DeviceInfo, DeviceStatus, DeviceType,
    FindingMeasurements, FindingType, ImagingParameters, MobileOptimizationRequest,
    MobileOptimizationResponse, NetworkConditions, OperatorLevel, PerformancePredictions,
    PerformanceTargets, PowerEstimates, PowerSettings, ProcessingConfig, ProcessingMetrics,
    QualityIndicators, RecommendationType, TissueCharacterization, TissueProperties, TissueRegion,
    UltrasoundDevice, UltrasoundFrame, UrgencyLevel,
};
