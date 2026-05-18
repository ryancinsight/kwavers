use std::io::Error as IoError;

use anyhow::anyhow;

use crate::core::error::{
    composite::MultiError, ConfigError, DataError, FieldError, GridError, KwaversError,
    MediumError, NumericalError, PhysicsError, SystemError, ValidationError,
};

use super::{
    metrics::SlidingRateWindow, AlertThreshold, ConsoleExporter, TelemetryErrorCounts, ErrorSeverity,
    TelemetryContext, TelemetryExporter,
};

#[test]
fn telemetry_context_creation() {
    let context = TelemetryContext::from_current_span();
    assert!(!context.trace_id.is_empty());
    assert!(!context.span_id.is_empty());
    assert_ne!(context.trace_id, context.span_id);
    assert_eq!(context.service_name, "kwavers");
}

#[test]
fn metrics_record_and_retrieve() {
    let metrics = TelemetryErrorCounts::new();
    let error = KwaversError::InternalError("test".to_string());

    metrics.record_error(&error);

    assert_eq!(metrics.error_count("critical"), 1);
    assert_eq!(metrics.error_count_by_type("internal"), 1);
    assert_eq!(metrics.total_errors(), 1);
}

#[test]
fn recovery_rate_calculation() {
    let metrics = TelemetryErrorCounts::new();

    for _ in 0..9 {
        metrics.record_recovery_attempt();
        metrics.record_recovery_success();
    }
    metrics.record_recovery_attempt();

    let rate = metrics.recovery_rate();
    assert!(rate > 0.89 && rate < 0.91);
}

#[test]
fn prometheus_export_format() {
    let metrics = TelemetryErrorCounts::new();
    let export = metrics.export_prometheus();

    assert!(export.contains("kwavers_errors_total"));
    assert!(export.contains("kwavers_errors_by_type_total"));
    assert!(export.contains("kwavers_recovery_success_rate"));
    assert!(export.contains("# HELP"));
    assert!(export.contains("# TYPE"));
}

#[test]
fn prometheus_export_includes_error_type_counter() {
    let metrics = TelemetryErrorCounts::new();
    let error = KwaversError::InternalError("test".to_string());

    metrics.record_error(&error);

    let export = metrics.export_prometheus();
    assert!(export.contains("error_type=\"internal\""));
}

#[test]
fn severity_thresholds() {
    assert!(ErrorSeverity::Critical.threshold() > ErrorSeverity::High.threshold());
    assert!(ErrorSeverity::High.threshold() > ErrorSeverity::Medium.threshold());
}

#[test]
fn alert_threshold_configuration() {
    let threshold = AlertThreshold {
        severity: ErrorSeverity::High,
        max_errors_per_minute: 5.0,
        consecutive_breaches: 3,
        window_seconds: 60,
    };

    assert_eq!(threshold.max_errors_per_minute, 5.0);
    assert_eq!(threshold.consecutive_breaches, 3);
}

#[test]
fn console_exporter_basic() {
    let exporter = ConsoleExporter;
    let context = TelemetryContext::from_current_span();
    let error = KwaversError::InternalError("test".to_string());
    exporter.export_error(&error, &context);
}

#[test]
fn severity_mapping_is_exhaustive_for_active_variants() {
    let cases = vec![
        (
            KwaversError::Grid(GridError::ZeroDimension {
                nx: 0,
                ny: 1,
                nz: 1,
            }),
            ErrorSeverity::High,
        ),
        (
            KwaversError::Medium(MediumError::InitializationFailed {
                reason: "init".to_string(),
            }),
            ErrorSeverity::High,
        ),
        (
            KwaversError::Physics(PhysicsError::InvalidParameter {
                parameter: "frequency".to_string(),
                value: -1.0,
                reason: "must be positive".to_string(),
            }),
            ErrorSeverity::High,
        ),
        (
            KwaversError::Data(DataError::FileNotFound {
                path: "missing.dat".to_string(),
            }),
            ErrorSeverity::Medium,
        ),
        (
            KwaversError::Config(ConfigError::MissingParameter {
                parameter: "dt".to_string(),
                section: "solver".to_string(),
            }),
            ErrorSeverity::Medium,
        ),
        (
            KwaversError::Numerical(NumericalError::DivisionByZero {
                operation: "test".to_string(),
                location: "unit".to_string(),
            }),
            ErrorSeverity::High,
        ),
        (
            KwaversError::Field(FieldError::NotRegistered("p".to_string())),
            ErrorSeverity::High,
        ),
        (
            KwaversError::System(SystemError::ResourceUnavailable {
                resource: "gpu".to_string(),
            }),
            ErrorSeverity::Critical,
        ),
        (
            KwaversError::Validation(ValidationError::OutOfRange {
                value: 10.0,
                min: 0.0,
                max: 1.0,
            }),
            ErrorSeverity::High,
        ),
        (
            KwaversError::InternalError("oops".to_string()),
            ErrorSeverity::Critical,
        ),
        (
            KwaversError::DimensionMismatch("bad dims".to_string()),
            ErrorSeverity::High,
        ),
        (
            KwaversError::FeatureNotAvailable("gpu".to_string()),
            ErrorSeverity::Low,
        ),
        (
            KwaversError::PerformanceError("slow".to_string()),
            ErrorSeverity::Low,
        ),
        (
            KwaversError::Io(IoError::other("io")),
            ErrorSeverity::Medium,
        ),
        (
            KwaversError::NotImplemented("missing".to_string()),
            ErrorSeverity::Low,
        ),
        (
            KwaversError::GpuError("OOM".to_string()),
            ErrorSeverity::Critical,
        ),
        (
            KwaversError::ResourceLimitExceeded {
                message: "mem".to_string(),
            },
            ErrorSeverity::Critical,
        ),
        (
            KwaversError::InvalidInput("bad".to_string()),
            ErrorSeverity::High,
        ),
        (
            KwaversError::MultipleErrors(MultiError::new()),
            ErrorSeverity::High,
        ),
        (
            KwaversError::ConcurrencyError {
                message: "lock".to_string(),
            },
            ErrorSeverity::Critical,
        ),
        (
            KwaversError::Visualization {
                message: "viz".to_string(),
            },
            ErrorSeverity::Low,
        ),
        (
            KwaversError::Shape(
                ndarray::Array2::<f64>::zeros((1, 1))
                    .into_shape_with_order((2,))
                    .unwrap_err(),
            ),
            ErrorSeverity::High,
        ),
        (KwaversError::Other(anyhow!("other")), ErrorSeverity::High),
    ];

    for (error, expected) in cases {
        assert_eq!(ErrorSeverity::from(&error), expected, "error={error}");
    }
}

#[test]
fn sliding_window_counts_recent_events_only() {
    let window = SlidingRateWindow::<60>::default();
    window.record_at(40);
    window.record_at(99);
    window.record_at(100);
    window.record_at(100);

    let rate = window.rate_per_minute_at(100, 60);
    assert_eq!(rate, 3.0);
}

#[test]
fn sliding_window_reuse_does_not_carry_stale_bucket_counts() {
    let window = SlidingRateWindow::<2>::default();
    for _ in 0..5 {
        window.record_at(1);
    }

    window.record_at(3);

    assert_eq!(window.rate_per_minute_at(3, 1), 60.0);
    assert_eq!(window.rate_per_minute_at(3, 2), 30.0);
}

#[test]
fn metrics_export_live_error_rate() {
    let metrics = TelemetryErrorCounts::new();
    let error = KwaversError::InternalError("test".to_string());

    metrics.record_error_with_time(&error, 139);
    metrics.record_error_with_time(&error, 200);
    metrics.record_error_with_time(&error, 200);

    let rate = metrics.error_rate_per_minute_at(ErrorSeverity::Critical, 200, 60);
    assert_eq!(rate, 2.0);
}

#[test]
fn telemetry_context_round_trips_span_identity() {
    let context = TelemetryContext::from_current_span();
    let span = context.as_span();

    assert_eq!(span.trace_id, context.trace_id);
    assert_eq!(span.span_id, context.span_id);
    assert_eq!(span.parent_span_id, context.parent_span_id);
    assert_eq!(span.service_name, context.service_name);
}
