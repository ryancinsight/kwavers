//! API middleware for authentication, rate limiting, and monitoring
//!
//! This module provides middleware components for the PINN API including
//! authentication, authorization, rate limiting, logging, and metrics collection.

use crate::infra::api::auth::AuthMiddleware;
use crate::infra::api::rate_limiter::{RateLimitConfig, RateLimiter};
use crate::infra::api::{APIError, APIErrorType, RateLimitInfo};
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use std::time::Instant;

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

            // Add rate limit headers
            let limit_info = rate_limiter.get_limit_info(&user_id, &endpoint);
            response.headers_mut().insert(
                "X-RateLimit-Limit",
                limit_info.limit.to_string().parse().unwrap(),
            );
            response.headers_mut().insert(
                "X-RateLimit-Remaining",
                limit_info.remaining.to_string().parse().unwrap(),
            );
            response.headers_mut().insert(
                "X-RateLimit-Reset",
                limit_info
                    .reset_time
                    .timestamp()
                    .to_string()
                    .parse()
                    .unwrap(),
            );

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

/// Production metrics collector using Prometheus and OpenTelemetry
#[derive(Clone, Debug)]
pub struct MetricsCollector {
    /// Prometheus registry for metrics
    registry: prometheus::Registry,
    /// HTTP request counter
    http_requests_total: prometheus::CounterVec,
    /// HTTP request duration histogram
    http_request_duration: prometheus::HistogramVec,
    /// Training job counter
    training_jobs_total: prometheus::CounterVec,
    /// Training job duration histogram
    training_job_duration: prometheus::Histogram,
    /// Inference request counter
    inference_requests_total: prometheus::CounterVec,
    /// Inference latency histogram
    inference_latency: prometheus::Histogram,
    /// Active connections gauge
    active_connections: prometheus::Gauge,
    /// Memory usage gauge
    memory_usage: prometheus::Gauge,
    /// GPU utilization gauge
    gpu_utilization: prometheus::Gauge,
}

impl MetricsCollector {
    /// Create a new production metrics collector
    pub fn new() -> Self {
        let registry = prometheus::Registry::new();

        // HTTP request metrics
        let http_requests_total = prometheus::CounterVec::new(
            prometheus::Opts::new("http_requests_total", "Total number of HTTP requests")
                .namespace("kwavers")
                .subsystem("api"),
            &["method", "endpoint", "status"],
        )
        .unwrap();

        let http_request_duration = prometheus::HistogramVec::new(
            prometheus::HistogramOpts::new(
                "http_request_duration_seconds",
                "HTTP request duration in seconds",
            )
            .namespace("kwavers")
            .subsystem("api")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]),
            &["method", "endpoint"],
        )
        .unwrap();

        // Training job metrics
        let training_jobs_total = prometheus::CounterVec::new(
            prometheus::Opts::new("training_jobs_total", "Total number of training jobs")
                .namespace("kwavers")
                .subsystem("ml"),
            &["status", "model_type"],
        )
        .unwrap();

        let training_job_duration = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "training_job_duration_seconds",
                "Training job duration in seconds",
            )
            .namespace("kwavers")
            .subsystem("ml")
            .buckets(vec![
                1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0,
            ]),
        )
        .unwrap();

        // Inference metrics
        let inference_requests_total = prometheus::CounterVec::new(
            prometheus::Opts::new(
                "inference_requests_total",
                "Total number of inference requests",
            )
            .namespace("kwavers")
            .subsystem("ml"),
            &["model_type", "status"],
        )
        .unwrap();

        let inference_latency = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "inference_latency_seconds",
                "Inference request latency in seconds",
            )
            .namespace("kwavers")
            .subsystem("ml")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
        )
        .unwrap();

        // System metrics
        let active_connections =
            prometheus::Gauge::new("active_connections", "Number of active connections").unwrap();

        let memory_usage =
            prometheus::Gauge::new("memory_usage_bytes", "Current memory usage in bytes").unwrap();

        let gpu_utilization =
            prometheus::Gauge::new("gpu_utilization_percent", "GPU utilization percentage")
                .unwrap();

        // Register all metrics
        registry
            .register(Box::new(http_requests_total.clone()))
            .unwrap();
        registry
            .register(Box::new(http_request_duration.clone()))
            .unwrap();
        registry
            .register(Box::new(training_jobs_total.clone()))
            .unwrap();
        registry
            .register(Box::new(training_job_duration.clone()))
            .unwrap();
        registry
            .register(Box::new(inference_requests_total.clone()))
            .unwrap();
        registry
            .register(Box::new(inference_latency.clone()))
            .unwrap();
        registry
            .register(Box::new(active_connections.clone()))
            .unwrap();
        registry.register(Box::new(memory_usage.clone())).unwrap();
        registry
            .register(Box::new(gpu_utilization.clone()))
            .unwrap();

        Self {
            registry,
            http_requests_total,
            http_request_duration,
            training_jobs_total,
            training_job_duration,
            inference_requests_total,
            inference_latency,
            active_connections,
            memory_usage,
            gpu_utilization,
        }
    }

    /// Record an HTTP request
    pub async fn record_request(&self, method: &str, status: u16, duration: std::time::Duration) {
        let status_str = status.to_string();
        let endpoint = "api"; // Could be made more specific

        self.http_requests_total
            .with_label_values(&[method, endpoint, &status_str])
            .inc();

        self.http_request_duration
            .with_label_values(&[method, endpoint])
            .observe(duration.as_secs_f64());
    }

    /// Record a training job completion
    pub async fn record_training_job(&self, duration_seconds: f64, status: &str, model_type: &str) {
        self.training_jobs_total
            .with_label_values(&[status, model_type])
            .inc();

        self.training_job_duration.observe(duration_seconds);
    }

    /// Record an inference request
    pub async fn record_inference_request(
        &self,
        latency_seconds: f64,
        model_type: &str,
        status: &str,
    ) {
        self.inference_requests_total
            .with_label_values(&[model_type, status])
            .inc();

        self.inference_latency.observe(latency_seconds);
    }

    /// Update active connections count
    pub async fn update_active_connections(&self, count: f64) {
        self.active_connections.set(count);
    }

    /// Update memory usage
    pub async fn update_memory_usage(&self, bytes: f64) {
        self.memory_usage.set(bytes);
    }

    /// Update GPU utilization
    pub async fn update_gpu_utilization(&self, percentage: f64) {
        self.gpu_utilization.set(percentage);
    }

    /// Get metrics in Prometheus format
    pub fn gather_metrics(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    /// Get the Prometheus registry for external access
    pub fn registry(&self) -> &prometheus::Registry {
        &self.registry
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

    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new();
        assert!(limiter.check_limit("user_123", "/api/train").await.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let metrics = MetricsCollector::new();
        metrics
            .record_request("GET", 200, std::time::Duration::from_millis(100))
            .await;
    }

    #[test]
    fn test_cors_middleware_creation() {
        let _cors = cors_middleware();
        // Middleware created successfully
    }
}
