//! API request handlers for PINN operations
//!
//! This module contains the HTTP request handlers for the enterprise PINN API,
//! implementing RESTful endpoints for training, inference, and model management.

use crate::infra::api::auth::AuthenticatedUser;
use crate::infra::api::{
    APIError, HealthCheck, JobInfoResponse, ListModelsResponse, ModelMetadata,
    PINNInferenceRequest, PINNInferenceResponse, PINNTrainingRequest, PINNTrainingResponse,
    PaginationParams,
};
use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::Json as JsonResponse,
};

// Explicit re-exports of clinical handlers for router setup
#[cfg(feature = "pinn")]
pub use crate::infra::api::clinical_handlers::{
    analyze_clinical, dicom_integrate, get_device_status, get_session_status, list_devices,
    optimize_mobile, register_device,
};

/// Application state shared across handlers
#[derive(Debug, Clone)]
pub struct AppState {
    /// Service version information
    pub version: String,
    /// Uptime tracking
    pub start_time: std::time::Instant,
    /// Job manager for PINN training operations
    pub job_manager: std::sync::Arc<crate::api::job_manager::JobManager>,
    /// Model registry for PINN model storage
    pub model_registry: std::sync::Arc<crate::api::model_registry::ModelRegistry>,
    /// Authentication middleware
    pub auth_middleware: std::sync::Arc<crate::api::auth::AuthMiddleware>,
}

/// Health check endpoint
#[axum::debug_handler]
pub async fn health_check(State(state): State<AppState>) -> JsonResponse<HealthCheck> {
    // In a real implementation, this would check database, cache, and queue health
    let uptime_seconds = state.start_time.elapsed().as_secs();

    let health = HealthCheck {
        status: crate::api::HealthStatus::Healthy,
        version: crate::api::APIVersion {
            version: state.version.clone(),
            build_date: std::env::var("VERGEN_BUILD_DATE")
                .unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("VERGEN_GIT_SHA").unwrap_or_else(|_| "unknown".to_string()),
        },
        uptime_seconds,
        database: crate::api::ServiceStatus::default(),
        cache: crate::api::ServiceStatus::default(),
        queue: crate::api::ServiceStatus::default(),
    };

    JsonResponse(health)
}

/// Train PINN model endpoint
#[axum::debug_handler]
pub async fn train_pinn_model(
    State(state): State<AppState>,
    auth: AuthenticatedUser,
    Json(request): Json<PINNTrainingRequest>,
) -> Result<JsonResponse<PINNTrainingResponse>, (StatusCode, JsonResponse<APIError>)> {
    // Submit training job to job manager
    match state.job_manager.submit_job(&auth.user_id, request).await {
        Ok(job_id) => {
            // Get the job to return current status
            if let Some(job) = state.job_manager.get_job(&job_id) {
                let response = PINNTrainingResponse {
                    job_id,
                    status: job.status,
                    estimated_completion: job.created_at + chrono::Duration::minutes(30), // Estimate
                    progress: job.progress,
                };
                Ok(JsonResponse(response))
            } else {
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    JsonResponse(APIError {
                        error: crate::api::APIErrorType::InternalError,
                        message: "Job submitted but not found".to_string(),
                        details: None,
                    }),
                ))
            }
        }
        Err(error) => Err((StatusCode::INTERNAL_SERVER_ERROR, JsonResponse(error))),
    }
}

/// Get job information endpoint
#[axum::debug_handler]
pub async fn get_job_info(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    auth: AuthenticatedUser,
) -> Result<JsonResponse<JobInfoResponse>, (StatusCode, JsonResponse<APIError>)> {
    // Get job from job manager
    if let Some(job) = state.job_manager.get_job(&job_id) {
        // Check if user owns this job
        if job.user_id != auth.user_id {
            return Err((
                StatusCode::FORBIDDEN,
                JsonResponse(APIError {
                    error: crate::api::APIErrorType::AuthorizationFailed,
                    message: "Not authorized to access this job".to_string(),
                    details: None,
                }),
            ));
        }

        let response = JobInfoResponse {
            job_id: job.id,
            status: job.status,
            created_at: job.created_at,
            started_at: job.started_at,
            completed_at: job.completed_at,
            progress: job.progress,
            result_url: job
                .result
                .as_ref()
                .map(|_| format!("/api/jobs/{}/result", job_id)), // Placeholder URL
            error_message: job.error_message,
        };

        Ok(JsonResponse(response))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            JsonResponse(APIError {
                error: crate::api::APIErrorType::ResourceNotFound,
                message: "Job not found".to_string(),
                details: None,
            }),
        ))
    }
}

/// Run PINN inference endpoint
#[axum::debug_handler]
pub async fn run_inference(
    State(state): State<AppState>,
    auth: AuthenticatedUser,
    Json(request): Json<PINNInferenceRequest>,
) -> Result<JsonResponse<PINNInferenceResponse>, (StatusCode, JsonResponse<APIError>)> {
    let _start_time = std::time::Instant::now();

    // Get model from registry
    let stored_model = match state.model_registry.get_model(&request.model_id) {
        Some(model) => model,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                JsonResponse(APIError {
                    error: crate::api::APIErrorType::ResourceNotFound,
                    message: format!("Model '{}' not found", request.model_id),
                    details: None,
                }),
            ));
        }
    };

    // Check if user owns this model
    if !state
        .model_registry
        .get_user_models(&auth.user_id)
        .iter()
        .any(|m| m.model_id == request.model_id)
    {
        return Err((
            StatusCode::FORBIDDEN,
            JsonResponse(APIError {
                error: crate::api::APIErrorType::AuthorizationFailed,
                message: "Not authorized to use this model".to_string(),
                details: None,
            }),
        ));
    }

    #[cfg(feature = "pinn")]
    {
        let _ = stored_model;
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            JsonResponse(APIError {
                error: crate::api::APIErrorType::ServiceUnavailable,
                message: "PINN inference is not available: stored model artifacts are not yet deserializable into an executable model".to_string(),
                details: None,
            }),
        ))
    }

    #[cfg(not(feature = "pinn"))]
    Err((
        StatusCode::INTERNAL_SERVER_ERROR,
        JsonResponse(APIError {
            error: crate::api::APIErrorType::InternalError,
            message: "PINN inference not available - feature not enabled".to_string(),
            details: None,
        }),
    ))
}

/// List available models endpoint
#[axum::debug_handler]
pub async fn list_models(
    State(state): State<AppState>,
    auth: AuthenticatedUser,
    Query(pagination): Query<PaginationParams>,
) -> Result<JsonResponse<ListModelsResponse>, (StatusCode, JsonResponse<APIError>)> {
    // Get user's models from registry
    let all_models = state.model_registry.get_user_models(&auth.user_id);

    // Apply pagination
    let page = pagination.page.unwrap_or(1).max(1);
    let page_size = pagination.page_size.unwrap_or(50).min(100); // Cap at 100
    let start_idx = (page - 1) * page_size;
    let end_idx = start_idx + page_size;

    let paginated_models = if start_idx < all_models.len() {
        all_models[start_idx..end_idx.min(all_models.len())].to_vec()
    } else {
        Vec::new()
    };

    let response = ListModelsResponse {
        models: paginated_models,
        total_count: all_models.len(),
        page,
        page_size,
    };

    Ok(JsonResponse(response))
}

/// Get model metadata endpoint
#[axum::debug_handler]
pub async fn get_model_info(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    auth: AuthenticatedUser,
) -> Result<JsonResponse<ModelMetadata>, (StatusCode, JsonResponse<APIError>)> {
    // Check if user owns this model
    let user_models = state.model_registry.get_user_models(&auth.user_id);
    if let Some(model_metadata) = user_models.iter().find(|m| m.model_id == model_id) {
        Ok(JsonResponse(model_metadata.clone()))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            JsonResponse(APIError {
                error: crate::api::APIErrorType::ResourceNotFound,
                message: "Model not found or access denied".to_string(),
                details: None,
            }),
        ))
    }
}

/// Delete model endpoint
#[axum::debug_handler]
pub async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    auth: AuthenticatedUser,
) -> Result<StatusCode, (StatusCode, JsonResponse<APIError>)> {
    // Delete model from registry
    match state.model_registry.delete_model(&auth.user_id, &model_id) {
        Ok(_) => Ok(StatusCode::NO_CONTENT),
        Err(error) => Err((StatusCode::INTERNAL_SERVER_ERROR, JsonResponse(error))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct DummyExecutor;

    impl crate::api::job_manager::TrainingExecutor for DummyExecutor {
        fn execute(
            &self,
            _request: crate::api::PINNTrainingRequest,
            _progress_sender: tokio::sync::mpsc::Sender<crate::api::TrainingProgress>,
        ) -> crate::api::job_manager::TrainingFuture {
            Box::pin(async {
                Err(crate::api::APIError {
                    error: crate::api::APIErrorType::InternalError,
                    message: "Dummy executor".to_string(),
                    details: None,
                })
            })
        }
    }

    #[tokio::test]
    async fn test_health_check() {
        let state = AppState {
            version: "1.0.0".to_string(),
            start_time: std::time::Instant::now(),
            job_manager: std::sync::Arc::new(crate::api::job_manager::JobManager::new(
                1,
                std::sync::Arc::new(DummyExecutor),
            )),
            model_registry: std::sync::Arc::new(crate::api::model_registry::ModelRegistry::new()),
            auth_middleware: std::sync::Arc::new(
                crate::api::auth::AuthMiddleware::new(
                    "test-secret-do-not-use-in-production",
                    crate::api::auth::JWTConfig::default(),
                )
                .expect("test auth middleware construction must succeed"),
            ),
        };

        let response = health_check(State(state)).await;
        let health: HealthCheck = response.0;

        assert_eq!(health.version.version, "1.0.0");
        assert!(matches!(health.status, crate::api::HealthStatus::Healthy));
    }

    #[test]
    fn test_pagination_defaults() {
        let pagination = PaginationParams::default();
        assert_eq!(pagination.page, Some(1));
        assert_eq!(pagination.page_size, Some(50));
    }
}
