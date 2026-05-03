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
