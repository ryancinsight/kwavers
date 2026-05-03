use std::time::SystemTime;

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable real-time quality monitoring
    pub enable_quality_monitoring: bool,
    /// Enable safety event logging
    pub enable_safety_logging: bool,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    /// History window size (frames)
    pub history_window: usize,
    /// Alert threshold for quality metrics
    pub quality_alert_threshold: f64,
    /// Alert threshold for safety parameters
    pub safety_alert_threshold: f64,
    /// Maximum allowed temperature rise (°C)
    pub max_temperature_rise_c: f64,
    /// Maximum allowed mechanical index
    pub max_mechanical_index: f64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_quality_monitoring: true,
            enable_safety_logging: true,
            enable_performance_profiling: true,
            history_window: 100,
            quality_alert_threshold: 0.7,
            safety_alert_threshold: 0.9,
            max_temperature_rise_c: 5.0,
            max_mechanical_index: 1.9,
        }
    }
}

/// Frame quality assessment
#[derive(Debug, Clone)]
pub struct FrameQualityRecord {
    /// Frame number in sequence
    pub frame_number: usize,
    /// Timestamp of frame capture
    pub timestamp: SystemTime,
    /// Processing time for this frame (ms)
    pub processing_time_ms: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
    /// Contrast (ratio of signal to background)
    pub contrast: f64,
    /// Spatial resolution estimate (mm)
    pub spatial_resolution_mm: f64,
    /// Artifact level (0-1, 0=clean, 1=severe)
    pub artifact_level: f64,
    /// Overall quality score (0-100)
    pub quality_score: f64,
}

/// Safety event log entry
#[derive(Debug, Clone)]
pub struct SafetyEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: SafetyEventType,
    /// Parameter value
    pub parameter_value: f64,
    /// Safety limit
    pub safety_limit: f64,
    /// Severity level
    pub severity: SafetySeverity,
    /// Human-readable description
    pub message: String,
}

/// Safety event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyEventType {
    /// Temperature exceeds limit
    TemperatureExceeded,
    /// Mechanical index exceeds limit
    MechanicalIndexExceeded,
    /// Dose limit approaching
    DoseApproaching,
    /// System resource warning
    ResourceWarning,
    /// Quality degradation
    QualityDegradation,
}

impl std::fmt::Display for SafetyEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TemperatureExceeded => write!(f, "Temperature Exceeded"),
            Self::MechanicalIndexExceeded => write!(f, "MI Exceeded"),
            Self::DoseApproaching => write!(f, "Dose Approaching"),
            Self::ResourceWarning => write!(f, "Resource Warning"),
            Self::QualityDegradation => write!(f, "Quality Degradation"),
        }
    }
}

/// Safety event severity
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum SafetySeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Urgent - immediate attention needed
    Urgent,
    /// Critical - system shutdown may be required
    Critical,
}

impl std::fmt::Display for SafetySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "Info"),
            Self::Warning => write!(f, "Warning"),
            Self::Urgent => write!(f, "Urgent"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// System performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total frames processed
    pub total_frames: usize,
    /// Frames with errors
    pub error_frames: usize,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Maximum processing time (ms)
    pub max_processing_time_ms: f64,
    /// Minimum processing time (ms)
    pub min_processing_time_ms: f64,
    /// Average frame rate (fps)
    pub avg_frame_rate_fps: f64,
    /// Uptime (seconds)
    pub uptime_seconds: f64,
}

/// Monitoring report
#[derive(Debug, Clone)]
pub struct MonitoringReport {
    /// Total uptime (seconds)
    pub uptime_seconds: f64,
    /// Total frames processed
    pub total_frames_processed: usize,
    /// Frames with errors
    pub error_frames: usize,
    /// Average frame rate (fps)
    pub avg_frame_rate_fps: f64,
    /// Average quality score (0-100)
    pub avg_quality_score: f64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Information events logged
    pub info_events: usize,
    /// Warning events logged
    pub warning_events: usize,
    /// Urgent events logged
    pub urgent_events: usize,
    /// Critical events logged
    pub critical_events: usize,
    /// Overall system status
    pub system_status: String,
}
