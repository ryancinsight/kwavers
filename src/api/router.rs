//! API Router Configuration
//!
//! This module configures the complete REST API router for both PINN operations
//! and clinical ultrasound integration, including middleware and route organization.

use crate::api::{
    handlers::{
        delete_model, get_job_info, get_model_info, health_check, list_models, run_inference,
        train_pinn_model, AppState,
    },
    middleware::MetricsCollector,
};
use axum::{
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    request_id::{MakeRequestUuid, SetRequestIdLayer},
    trace::TraceLayer,
};

#[cfg(feature = "pinn")]
use crate::api::clinical_handlers::{
    analyze_clinical, dicom_integrate, get_device_status, get_session_status, list_devices,
    optimize_mobile, register_device, ClinicalAppState,
};

/// Create the complete API router
pub fn create_router() -> Router<()> {
    // Create application states
    // Correctness + security invariant:
    // Router construction MUST NOT rely on `AuthMiddleware::default()`, because that allows
    // accidental deployments with an implicit/placeholder secret. We require an explicit secret
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
        job_manager: std::sync::Arc::new(crate::api::job_manager::JobManager::new(5)),
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
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn test_router_creation() {
        // `create_router()` requires an explicit secret via KWAVERS_JWT_SECRET.
        std::env::set_var("KWAVERS_JWT_SECRET", "test-secret-do-not-use-in-production");

        let router = create_router();

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
