//! API Router Configuration
//!
//! This module configures the complete REST API router for both PINN operations
//! and clinical ultrasound integration, including middleware and route organization.

use crate::infrastructure::api::handlers::{
    delete_model, get_job_info, get_model_info, health_check, list_models, run_inference,
    train_pinn_model, AppState,
};
use crate::infrastructure::api::job_manager::TrainingExecutor;
use axum::{
    routing::{delete, get, post},
    Router,
};
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    request_id::SetRequestIdLayer,
    trace::TraceLayer,
};

#[cfg(feature = "pinn")]
use crate::infrastructure::api::clinical_handlers::{
    analyze_clinical, dicom_integrate, get_device_status, get_session_status, list_devices,
    optimize_mobile, register_device, ClinicalAppState,
};

/// Create the complete API router
pub fn create_router(training_executor: std::sync::Arc<dyn TrainingExecutor>) -> Router<()> {
    // Create application states
    // Correctness + security invariant:
    // Router construction MUST NOT rely on `AuthMiddleware::default()`, because that allows
    // accidental deployments with an implicit/placeholder secret. We require an explicit secret
    // TODO_AUDIT: P2 - Production API Architecture - Implement complete REST/GraphQL API with security, monitoring, and scalability
    // DEPENDS ON: infra/api/graphql.rs, infra/api/security.rs, infra/api/rate_limiting.rs, infra/api/caching.rs
    // MISSING: GraphQL federation for microservice architecture
    // MISSING: OAuth 2.0 / OpenID Connect authentication
    // MISSING: Rate limiting with token bucket algorithms
    // MISSING: API versioning and backward compatibility
    // MISSING: Response caching with Redis/CDN integration
    // MISSING: API documentation generation (OpenAPI/Swagger)
    // MISSING: Request validation with JSON Schema
    // THEOREM: Little's law: L = λW for queueing systems (concurrency = arrival_rate × service_time)
    // THEOREM: Brewer CAP theorem: Choose 2 of Consistency, Availability, Partition tolerance
    // REFERENCES: Fielding (2000) Architectural Styles; Richardson (2013) Microservices Architecture
    // sourced from the process environment (or secret manager wiring upstream).
    //
    // For tests, construct `AppState` directly in test modules with an explicit test-secret.
    let jwt_secret = std::env::var("KWAVERS_JWT_SECRET").expect(
        "KWAVERS_JWT_SECRET must be set to a strong, non-empty secret to construct the API router",
    );
    let jwt_secret = jwt_secret.trim();
    assert!(
        !jwt_secret.is_empty(),
        "KWAVERS_JWT_SECRET is set but empty/whitespace; set it to a strong, non-empty secret"
    );
    assert!(
        jwt_secret != "default-secret-change-in-production",
        "KWAVERS_JWT_SECRET is set to a known placeholder; set it to a strong, unique secret"
    );

    let pinn_state = AppState {
        version: env!("CARGO_PKG_VERSION").to_string(),
        start_time: std::time::Instant::now(),
        job_manager: std::sync::Arc::new(crate::api::job_manager::JobManager::new(
            5,
            training_executor,
        )),
        model_registry: std::sync::Arc::new(crate::api::model_registry::ModelRegistry::new()),
        auth_middleware: std::sync::Arc::new(
            crate::api::auth::AuthMiddleware::new(
                jwt_secret,
                crate::api::auth::JWTConfig::default(),
            )
            .expect("Failed to construct AuthMiddleware from KWAVERS_JWT_SECRET"),
        ),
    };

    // Build router with middleware stack
    #[allow(unused_mut)] // mut needed when `pinn` feature is enabled
    let mut router = Router::<AppState>::new()
        // Health and monitoring
        .route("/health", get(health_check))
        // PINN API routes (/api/pinn/*)
        .nest("/api/pinn", create_pinn_router());

    // Add clinical routes if PINN feature is enabled
    #[cfg(feature = "pinn")]
    {
        let clinical_state = crate::api::clinical_handlers::ClinicalAppState::new(
            pinn_state.auth_middleware.clone(),
        )
        .expect("Failed to create clinical app state");
        router = router.nest(
            "/api/clinical",
            create_clinical_router().with_state(clinical_state),
        );
    }

    router
        // Apply global middleware
        .layer(SetRequestIdLayer::new(
            axum::http::HeaderName::from_static("x-request-id"),
            tower_http::request_id::MakeRequestUuid,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(pinn_state)
}

/// Create PINN-specific routes
fn create_pinn_router() -> Router<AppState> {
    Router::new()
        // Training operations
        .route("/train", post(train_pinn_model))
        .route("/jobs/:job_id", get(get_job_info))
        // Inference operations
        .route("/inference", post(run_inference))
        // Model management
        .route("/models", get(list_models))
        .route("/models/:model_id", get(get_model_info))
        .route("/models/:model_id", delete(delete_model))
}

/// Create clinical ultrasound routes
#[cfg(feature = "pinn")]
fn create_clinical_router() -> Router<ClinicalAppState> {
    Router::new()
        // Device management
        .route("/devices", post(register_device))
        .route("/devices", get(list_devices))
        .route("/devices/:device_id", get(get_device_status))
        // Clinical analysis (AI-enhanced beamforming)
        .route("/analyze", post(analyze_clinical))
        // Standards integration
        .route("/dicom", post(dicom_integrate))
        // Mobile optimization
        .route("/optimize", post(optimize_mobile))
        // Session management
        .route("/sessions/:session_id", get(get_session_status))
}

/// Create development/test router with additional debugging endpoints
#[cfg(debug_assertions)]
pub fn create_dev_router() -> Router<()> {
    Router::new()
        // Add development-only routes here if needed
        .route("/debug/info", get(debug_info))
}

/// Debug information endpoint (development only)
#[cfg(debug_assertions)]
async fn debug_info() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "build_date": env!("CARGO_PKG_VERSION", "unknown"),
        "commit_hash": "unknown",
        "features": {
            "pinn": cfg!(feature = "pinn"),
            "gpu": cfg!(feature = "gpu"),
            "simd": cfg!(feature = "simd"),
        },
        "memory_info": {
            "allocated_mb": 0, // Would need memory tracking implementation
            "available_mb": 0,
        },
        "active_connections": 0, // Would need connection tracking
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tokio::sync::mpsc;
    use tower::util::ServiceExt;

    #[derive(Debug)]
    struct TestTrainingExecutor;

    impl TrainingExecutor for TestTrainingExecutor {
        fn execute(
            &self,
            _request: crate::infrastructure::api::PINNTrainingRequest,
            _progress_sender: mpsc::Sender<crate::infrastructure::api::TrainingProgress>,
        ) -> crate::infrastructure::api::job_manager::TrainingFuture {
            Box::pin(async move {
                Err(crate::infrastructure::api::APIError {
                    error: crate::infrastructure::api::APIErrorType::InternalError,
                    message: "Test training executor invoked".to_string(),
                    details: None,
                })
            })
        }
    }

    #[tokio::test]
    async fn test_router_creation() {
        // `create_router()` requires an explicit secret via KWAVERS_JWT_SECRET.
        std::env::set_var("KWAVERS_JWT_SECRET", "test-secret-do-not-use-in-production");

        let router = create_router(std::sync::Arc::new(TestTrainingExecutor));

        // Validate health endpoint responds OK
        let request: Request<Body> = Request::get("/health").body(Body::empty()).unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn test_dev_router_creation() {
        // `create_dev_router()` builds on `create_router()` and therefore requires an explicit secret.
        std::env::set_var("KWAVERS_JWT_SECRET", "test-secret-do-not-use-in-production");

        let router = create_dev_router();

        // Test debug endpoint
        let request: Request<Body> = Request::get("/debug/info").body(Body::empty()).unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    #[cfg(feature = "pinn")]
    #[tokio::test]
    async fn test_clinical_router_endpoints() {
        let clinical_state = ClinicalAppState::new(std::sync::Arc::new(
            crate::api::auth::AuthMiddleware::new(
                "test-secret-do-not-use-in-production",
                crate::api::auth::JWTConfig::default(),
            )
            .expect("test auth middleware construction must succeed"),
        ))
        .unwrap();
        let router = create_clinical_router().with_state(clinical_state);

        // Test that clinical devices endpoint responds OK
        let request: Request<Body> = Request::get("/devices").body(Body::empty()).unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }
}
