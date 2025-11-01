//! API middleware for authentication, rate limiting, and monitoring
//!
//! This module provides middleware components for the PINN API including
//! authentication, authorization, rate limiting, logging, and metrics collection.

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use std::time::Instant;
use crate::api::{APIError, APIErrorType, RateLimitInfo};
use crate::api::auth::AuthMiddleware;

/// Authentication middleware
pub async fn auth_middleware(
    State(auth_middleware): State<AuthMiddleware>,
    mut request: Request,
    next: Next,
) -> Result<Response, (StatusCode, String)> {
    // Extract token from Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    let token = if let Some(auth_header) = auth_header {
        if let Some(token) = auth_header.strip_prefix("Bearer ") {
            token
        } else {
            return Err((
                StatusCode::UNAUTHORIZED,
                "Invalid authorization header format".to_string(),
            ));
        }
    } else {
        return Err((
            StatusCode::UNAUTHORIZED,
            "Missing authorization header".to_string(),
        ));
    };

    // Authenticate user
    match auth_middleware.authenticate_jwt(token) {
        Ok(user) => {
            // Insert authenticated user into request extensions
            request.extensions_mut().insert(user);
            Ok(next.run(request).await)
        }
        Err(error) => {
            let status_code = match error.error {
                APIErrorType::AuthenticationFailed => StatusCode::UNAUTHORIZED,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            Err((status_code, error.message))
        }
    }
}

/// Optional authentication middleware (for endpoints that support both authenticated and anonymous access)
pub async fn optional_auth_middleware(
    State(auth_middleware): State<AuthMiddleware>,
    mut request: Request,
    next: Next,
) -> Response {
    // Try to extract and authenticate token
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));

    if let Some(token) = auth_header {
        if let Ok(user) = auth_middleware.authenticate_jwt(token) {
            request.extensions_mut().insert(user);
        }
    }

    next.run(request).await
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(_rate_limiter): State<RateLimiter>,
    request: Request,
    next: Next,
) -> Response {
    // TODO: Implement rate limiting logic
    // For now, allow all requests

    next.run(request).await
}

/// Logging and metrics middleware
pub async fn logging_middleware(
    State(_metrics): State<MetricsCollector>,
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    // Process request
    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    // Log request details
    tracing::info!(
        method = %method,
        uri = %uri,
        status = %status,
        duration_ms = duration.as_millis(),
        "API request completed"
    );

    // TODO: Record metrics
    // metrics.record_request(method.as_str(), status.as_u16(), duration);

    response
}

/// CORS middleware configuration
pub fn cors_middleware() -> tower_http::cors::CorsLayer {
    tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any)
        .allow_credentials(false)
}

/// Request ID middleware
pub fn request_id_middleware() -> tower_http::request_id::SetRequestIdLayer<tower_http::request_id::MakeRequestUuid> {
    tower_http::request_id::SetRequestIdLayer::x_request_id(tower_http::request_id::MakeRequestUuid)
}

/// Compression middleware
pub fn compression_middleware() -> tower_http::compression::CompressionLayer {
    tower_http::compression::CompressionLayer::new()
}

/// Tracing middleware
pub fn tracing_middleware() -> tower_http::trace::TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>
> {
    tower_http::trace::TraceLayer::new_for_http()
}

/// Rate limiter placeholder
#[derive(Clone, Debug)]
pub struct RateLimiter;

/// Metrics collector placeholder
#[derive(Clone, Debug)]
pub struct MetricsCollector;

impl RateLimiter {
    pub fn new() -> Self {
        Self
    }

    pub async fn check_limit(&self, _user_id: &str, _endpoint: &str) -> Result<(), APIError> {
        // TODO: Implement rate limiting
        Ok(())
    }

    pub fn get_limit_info(&self, _user_id: &str, _endpoint: &str) -> RateLimitInfo {
        // TODO: Return actual rate limit info
        RateLimitInfo {
            limit: 100,
            remaining: 95,
            reset_time: chrono::Utc::now() + chrono::Duration::minutes(1),
        }
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self
    }

    pub async fn record_request(&self, _method: &str, _status: u16, _duration: std::time::Duration) {
        // TODO: Record request metrics
    }

    pub async fn record_training_job(&self, _duration: f64) {
        // TODO: Record training job metrics
    }

    pub async fn record_inference_request(&self, _latency: f64) {
        // TODO: Record inference request metrics
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use tower::Service;

    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new();
        assert!(limiter.check_limit("user_123", "/api/train").await.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let metrics = MetricsCollector::new();
        metrics.record_request("GET", 200, std::time::Duration::from_millis(100)).await;
    }

    #[test]
    fn test_cors_middleware_creation() {
        let _cors = cors_middleware();
        // Middleware created successfully
    }
}
