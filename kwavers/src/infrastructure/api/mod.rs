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
    APIConfig, APIError, APIErrorType, APIVersion, ApiBoundaryConditionSpec, ApiModelMetadata,
    ApiRateLimitConfig, GeometrySpec, HealthCheck, HealthStatus, JobInfoResponse, JobStatus,
    ListModelsResponse, ObstacleSpec, PINNInferenceRequest, PINNInferenceResponse,
    PINNTrainingRequest, PINNTrainingResponse, PaginationParams, PinnApiPhysicsParameters,
    PinnApiTrainingConfig, PinnApiTrainingMetrics, RateLimitInfo, ServiceStatus, TrainingProgress,
};

// Clinical ultrasound API type re-exports.
#[cfg(feature = "pinn")]
pub use models::{
    AbnormalRegion, AnalysisPriority, ApiDeviceInfo, ApiDeviceStatus, ApiTissueProperties,
    ApiTissueRegion, ClinicalAnalysisRequest, ClinicalAnalysisResponse, ClinicalContext,
    ClinicalFinding, ClinicalRecommendation, ConnectionType, DICOMIntegrationRequest,
    DICOMIntegrationResponse, DICOMStudyInfo, DICOMValue, DeviceCapabilities, DeviceCapability,
    DeviceType, FindingMeasurements, FindingType, ImagingParameters, MobileOptimizationRequest,
    MobileOptimizationResponse, MobileProcessingConfig, NetworkConditions, OperatorLevel,
    PerformancePredictions, PerformanceTargets, PowerEstimates, PowerSettings, ProcessingMetrics,
    QualityIndicators, RecommendationType, TissueCharacterization, UltrasoundDevice,
    UltrasoundFrame, UrgencyLevel,
};
