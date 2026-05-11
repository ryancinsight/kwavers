//! API middleware for authentication, rate limiting, and monitoring
//!
//! This module provides middleware components for the PINN API including
//! authentication, authorization, rate limiting, logging, and metrics collection.

use crate::infrastructure::api::auth::AuthMiddleware;
use crate::infrastructure::api::rate_limiter::RateLimiter;
use crate::infrastructure::api::APIErrorType;
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use std::time::Instant;

mod metrics;
pub use metrics::MetricsCollector;

#[cfg(test)]
mod tests;

/// Authentication middleware
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
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
    State(rate_limiter): State<RateLimiter>,
    request: Request,
    next: Next,
) -> Response {
    // Extract user ID from request extensions (set by auth middleware)
    let user_id = request
        .extensions()
        .get::<crate::api::auth::AuthenticatedUser>()
        .map(|user| user.user_id.clone())
        .unwrap_or_else(|| "anonymous".to_string());

    // Extract endpoint path
    let endpoint = request.uri().path().to_string();

    // Check rate limit
    match rate_limiter.check_limit(&user_id, &endpoint).await {
        Ok(_) => next.run(request).await,
        Err(error) => {
            // Return rate limit exceeded response
            let mut response = axum::response::Response::new(axum::body::Body::from(
                serde_json::to_string(&error).unwrap_or_default(),
            ));
            *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;

            // Add rate limit headers (to_string() on integers always produces
            // valid ASCII, so HeaderValue::from_str cannot fail here).
            let limit_info = rate_limiter.get_limit_info(&user_id, &endpoint);
            if let Ok(v) = limit_info.limit.to_string().parse() {
                response.headers_mut().insert("X-RateLimit-Limit", v);
            }
            if let Ok(v) = limit_info.remaining.to_string().parse() {
                response.headers_mut().insert("X-RateLimit-Remaining", v);
            }
            if let Ok(v) = limit_info.reset_time.timestamp().to_string().parse() {
                response.headers_mut().insert("X-RateLimit-Reset", v);
            }

            response
        }
    }
}

/// Logging and metrics middleware
pub async fn logging_middleware(
    State(metrics): State<MetricsCollector>,
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

    // Record metrics
    metrics
        .record_request(method.as_str(), status.as_u16(), duration)
        .await;

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
pub fn request_id_middleware(
) -> tower_http::request_id::SetRequestIdLayer<tower_http::request_id::MakeRequestUuid> {
    tower_http::request_id::SetRequestIdLayer::x_request_id(tower_http::request_id::MakeRequestUuid)
}

/// Compression middleware
pub fn compression_middleware() -> tower_http::compression::CompressionLayer {
    tower_http::compression::CompressionLayer::new()
}

/// Tracing middleware
pub fn tracing_middleware() -> tower_http::trace::TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
> {
    tower_http::trace::TraceLayer::new_for_http()
}
