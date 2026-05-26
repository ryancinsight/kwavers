//! Production metrics collector using Prometheus and OpenTelemetry.

use crate::core::constants::numerical::{SECONDS_PER_HOUR, SECONDS_PER_MINUTE};

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
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
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
                1.0,
                5.0,
                10.0,
                30.0,
                SECONDS_PER_MINUTE,
                5.0 * SECONDS_PER_MINUTE,
                10.0 * SECONDS_PER_MINUTE,
                30.0 * SECONDS_PER_MINUTE,
                SECONDS_PER_HOUR,
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
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
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
