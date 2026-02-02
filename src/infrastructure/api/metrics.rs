//! Prometheus metrics collection for PINN API
//!
//! This module provides comprehensive metrics collection and exposure for monitoring
//! the PINN API performance, training jobs, and system health using Prometheus.

use lazy_static::lazy_static;
use prometheus::{
    register_counter, register_gauge, register_histogram, register_int_counter, register_int_gauge,
    Counter, Encoder, Gauge, Histogram, IntCounter, IntGauge, TextEncoder,
};
use std::collections::HashMap;

/// HTTP request metrics
lazy_static! {
    pub static ref HTTP_REQUESTS_TOTAL: IntCounter =
        register_int_counter!("http_requests_total", "Total number of HTTP requests")
            .expect("Can't create HTTP_REQUESTS_TOTAL metric");

    pub static ref HTTP_REQUEST_DURATION: Histogram =
        register_histogram!("http_request_duration_seconds", "HTTP request duration in seconds")
            .expect("Can't create HTTP_REQUEST_DURATION metric");

    pub static ref HTTP_REQUEST_SIZE: Histogram =
        register_histogram!("http_request_size_bytes", "HTTP request size in bytes")
            .expect("Can't create HTTP_REQUEST_SIZE metric");

    pub static ref HTTP_RESPONSE_SIZE: Histogram =
        register_histogram!("http_response_size_bytes", "HTTP response size in bytes")
            .expect("Can't create HTTP_RESPONSE_SIZE metric");
}

/// PINN training job metrics
lazy_static! {
    pub static ref PINN_TRAINING_JOBS_TOTAL: IntCounter =
        register_int_counter!("pinn_training_jobs_total", "Total number of training jobs submitted")
            .expect("Can't create PINN_TRAINING_JOBS_TOTAL metric");

    pub static ref PINN_TRAINING_JOBS_ACTIVE: IntGauge =
        register_int_gauge!("pinn_training_jobs_active", "Number of currently active training jobs")
            .expect("Can't create PINN_TRAINING_JOBS_ACTIVE metric");

    pub static ref PINN_TRAINING_JOBS_COMPLETED: IntCounter =
        register_int_counter!("pinn_training_jobs_completed_total", "Total number of completed training jobs")
            .expect("Can't create PINN_TRAINING_JOBS_COMPLETED metric");

    pub static ref PINN_TRAINING_JOBS_FAILED: IntCounter =
        register_int_counter!("pinn_training_jobs_failed_total", "Total number of failed training jobs")
            .expect("Can't create PINN_TRAINING_JOBS_FAILED metric");

    pub static ref PINN_TRAINING_DURATION: Histogram =
        register_histogram!("pinn_training_duration_seconds", "Training job duration in seconds")
            .expect("Can't create PINN_TRAINING_DURATION metric");

    pub static ref PINN_TRAINING_FINAL_LOSS: Histogram =
        register_histogram!("pinn_training_final_loss", "Final loss values of completed training jobs")
            .expect("Can't create PINN_TRAINING_FINAL_LOSS metric");
}

/// PINN inference metrics
lazy_static! {
    pub static ref PINN_INFERENCE_REQUESTS_TOTAL: IntCounter =
        register_int_counter!("pinn_inference_requests_total", "Total number of inference requests")
            .expect("Can't create PINN_INFERENCE_REQUESTS_TOTAL metric");

    pub static ref PINN_INFERENCE_LATENCY: Histogram =
        register_histogram!("pinn_inference_latency_seconds", "Inference request latency in seconds")
            .expect("Can't create PINN_INFERENCE_LATENCY metric");

    pub static ref PINN_INFERENCE_BATCH_SIZE: Histogram =
        register_histogram!("pinn_inference_batch_size", "Inference batch sizes")
            .expect("Can't create PINN_INFERENCE_BATCH_SIZE metric");
}

/// System resource metrics
lazy_static! {
    pub static ref SYSTEM_CPU_USAGE: Gauge =
        register_gauge!("system_cpu_usage_percent", "System CPU usage percentage")
            .expect("Can't create SYSTEM_CPU_USAGE metric");

    pub static ref SYSTEM_MEMORY_USAGE: Gauge =
        register_gauge!("system_memory_usage_bytes", "System memory usage in bytes")
            .expect("Can't create SYSTEM_MEMORY_USAGE metric");

    pub static ref SYSTEM_GPU_MEMORY_USAGE: Gauge =
        register_gauge!("system_gpu_memory_usage_bytes", "GPU memory usage in bytes")
            .expect("Can't create SYSTEM_GPU_MEMORY_USAGE metric");

    pub static ref SYSTEM_DISK_USAGE: Gauge =
        register_gauge!("system_disk_usage_bytes", "Disk usage in bytes")
            .expect("Can't create SYSTEM_DISK_USAGE metric");
}

/// Database connection metrics
lazy_static! {
    pub static ref DATABASE_CONNECTIONS_ACTIVE: IntGauge =
        register_int_gauge!("database_connections_active", "Number of active database connections")
            .expect("Can't create DATABASE_CONNECTIONS_ACTIVE metric");

    pub static ref DATABASE_CONNECTIONS_IDLE: IntGauge =
        register_int_gauge!("database_connections_idle", "Number of idle database connections")
            .expect("Can't create DATABASE_CONNECTIONS_IDLE metric");

    pub static ref DATABASE_QUERY_DURATION: Histogram =
        register_histogram!("database_query_duration_seconds", "Database query duration in seconds")
            .expect("Can't create DATABASE_QUERY_DURATION metric");
}

/// Cache metrics
lazy_static! {
    pub static ref CACHE_HITS_TOTAL: IntCounter =
        register_int_counter!("cache_hits_total", "Total number of cache hits")
            .expect("Can't create CACHE_HITS_TOTAL metric");

    pub static ref CACHE_MISSES_TOTAL: IntCounter =
        register_int_counter!("cache_misses_total", "Total number of cache misses")
            .expect("Can't create CACHE_MISSES_TOTAL metric");

    pub static ref CACHE_SIZE: IntGauge =
        register_int_gauge!("cache_size", "Current cache size")
            .expect("Can't create CACHE_SIZE metric");
}

/// Error metrics
lazy_static! {
    pub static ref ERRORS_TOTAL: IntCounter =
        register_int_counter!("errors_total", "Total number of errors")
            .expect("Can't create ERRORS_TOTAL metric");
}

/// Rate limiting metrics
lazy_static! {
    pub static ref RATE_LIMIT_HITS_TOTAL: IntCounter =
        register_int_counter!("rate_limit_hits_total", "Total number of rate limit hits")
            .expect("Can't create RATE_LIMIT_HITS_TOTAL metric");

    pub static ref RATE_LIMIT_REQUESTS_TOTAL: IntCounter =
        register_int_counter!("rate_limit_requests_total", "Total number of rate limited requests")
            .expect("Can't create RATE_LIMIT_REQUESTS_TOTAL metric");
}

/// Metrics collector and exporter
pub struct MetricsCollector {
    encoder: TextEncoder,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            encoder: TextEncoder::new(),
        }
    }

    /// Export metrics in Prometheus format
    pub fn export(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        self.encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    /// Record HTTP request metrics
    pub fn record_http_request(&self, method: &str, status: u16, duration: std::time::Duration, request_size: usize, response_size: usize) {
        HTTP_REQUESTS_TOTAL.inc();
        HTTP_REQUEST_DURATION.observe(duration.as_secs_f64());
        HTTP_REQUEST_SIZE.observe(request_size as f64);
        HTTP_RESPONSE_SIZE.observe(response_size as f64);

        // Add custom labels if needed
        // Note: Prometheus counters don't support labels directly,
        // we'd need to create separate metrics for each combination
    }

    /// Record training job started
    pub fn record_training_job_started(&self) {
        PINN_TRAINING_JOBS_TOTAL.inc();
        PINN_TRAINING_JOBS_ACTIVE.inc();
    }

    /// Record training job completed
    pub fn record_training_job_completed(&self, duration: f64, final_loss: f64) {
        PINN_TRAINING_JOBS_ACTIVE.dec();
        PINN_TRAINING_JOBS_COMPLETED.inc();
        PINN_TRAINING_DURATION.observe(duration);
        PINN_TRAINING_FINAL_LOSS.observe(final_loss);
    }

    /// Record training job failed
    pub fn record_training_job_failed(&self) {
        PINN_TRAINING_JOBS_ACTIVE.dec();
        PINN_TRAINING_JOBS_FAILED.inc();
    }

    /// Record inference request
    pub fn record_inference_request(&self, latency: f64, batch_size: usize) {
        PINN_INFERENCE_REQUESTS_TOTAL.inc();
        PINN_INFERENCE_LATENCY.observe(latency);
        PINN_INFERENCE_BATCH_SIZE.observe(batch_size as f64);
    }

    /// Update system metrics
    pub fn update_system_metrics(&self, cpu_usage: f64, memory_usage: u64, gpu_memory_usage: Option<u64>, disk_usage: u64) {
        SYSTEM_CPU_USAGE.set(cpu_usage);
        SYSTEM_MEMORY_USAGE.set(memory_usage as f64);
        if let Some(gpu_mem) = gpu_memory_usage {
            SYSTEM_GPU_MEMORY_USAGE.set(gpu_mem as f64);
        }
        SYSTEM_DISK_USAGE.set(disk_usage as f64);
    }

    /// Update database metrics
    pub fn update_database_metrics(&self, active_connections: i64, idle_connections: i64) {
        DATABASE_CONNECTIONS_ACTIVE.set(active_connections);
        DATABASE_CONNECTIONS_IDLE.set(idle_connections);
    }

    /// Record database query
    pub fn record_database_query(&self, duration: f64) {
        DATABASE_QUERY_DURATION.observe(duration);
    }

    /// Record cache operation
    pub fn record_cache_hit(&self) {
        CACHE_HITS_TOTAL.inc();
    }

    pub fn record_cache_miss(&self) {
        CACHE_MISSES_TOTAL.inc();
    }

    pub fn update_cache_size(&self, size: i64) {
        CACHE_SIZE.set(size);
    }

    /// Record error
    pub fn record_error(&self, error_type: &str) {
        ERRORS_TOTAL.inc();
        // Could add labeled error counters if needed
    }

    /// Record rate limiting
    pub fn record_rate_limit_hit(&self) {
        RATE_LIMIT_HITS_TOTAL.inc();
    }

    pub fn record_rate_limited_request(&self) {
        RATE_LIMIT_REQUESTS_TOTAL.inc();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check metrics
pub struct HealthMetrics {
    pub database_healthy: bool,
    pub cache_healthy: bool,
    pub gpu_available: bool,
    pub last_health_check: std::time::Instant,
}

impl HealthMetrics {
    pub fn new() -> Self {
        Self {
            database_healthy: false,
            cache_healthy: false,
            gpu_available: false,
            last_health_check: std::time::Instant::now(),
        }
    }

    pub fn update_database_health(&mut self, healthy: bool) {
        self.database_healthy = healthy;
    }

    pub fn update_cache_health(&mut self, healthy: bool) {
        self.cache_healthy = healthy;
    }

    pub fn update_gpu_availability(&mut self, available: bool) {
        self.gpu_available = available;
    }

    pub fn is_healthy(&self) -> bool {
        self.database_healthy && self.cache_healthy
    }

    pub fn seconds_since_last_check(&self) -> u64 {
        self.last_health_check.elapsed().as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert!(collector.export().is_ok());
    }

    #[test]
    fn test_health_metrics() {
        let mut health = HealthMetrics::new();

        assert!(!health.is_healthy());

        health.update_database_health(true);
        health.update_cache_health(true);

        assert!(health.is_healthy());
    }

    #[test]
    fn test_training_job_metrics() {
        let collector = MetricsCollector::new();

        // Initially no active jobs
        collector.record_training_job_started();
        collector.record_training_job_started();

        // Complete one job
        collector.record_training_job_completed(120.5, 0.001);

        // Export should succeed
        assert!(collector.export().is_ok());
    }
}
