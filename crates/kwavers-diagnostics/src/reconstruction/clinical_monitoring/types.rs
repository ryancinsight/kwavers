use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, Length, TemperatureDifference, Time,
};
use aequitas::systems::si::units::Kelvin;
use kwavers_core::constants::medical::MI_LIMIT_SOFT_TISSUE;
use std::time::SystemTime;

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct ClinicalMonitoringConfig {
    /// Enable real-time quality monitoring
    pub enable_quality_monitoring: bool,
    /// Enable safety event logging
    pub enable_safety_logging: bool,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    /// History window size (frames)
    pub history_window: usize,
    /// Alert threshold for quality metrics
    pub quality_alert_threshold: Dimensionless,
    /// Alert threshold for safety parameters
    pub safety_alert_threshold: Dimensionless,
    /// Maximum allowed temperature rise.
    pub max_temperature_rise: TemperatureDifference,
    /// Maximum allowed mechanical index
    pub max_mechanical_index: Dimensionless,
}

impl Default for ClinicalMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_quality_monitoring: true,
            enable_safety_logging: true,
            enable_performance_profiling: true,
            history_window: 100,
            quality_alert_threshold: Dimensionless::from_base(0.7),
            safety_alert_threshold: Dimensionless::from_base(0.9),
            max_temperature_rise: TemperatureDifference::from_unit::<Kelvin>(5.0),
            max_mechanical_index: Dimensionless::from_base(MI_LIMIT_SOFT_TISSUE),
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
    /// Processing time for this frame.
    pub processing_time: Time,
    /// Signal-to-noise ratio, represented as a dimensionless logarithmic ratio.
    pub snr: Dimensionless,
    /// Contrast (ratio of signal to background).
    pub contrast: Dimensionless,
    /// Spatial resolution estimate.
    pub spatial_resolution: Length,
    /// Artifact level (0-1, 0=clean, 1=severe).
    pub artifact_level: Dimensionless,
    /// Overall quality score (0-100).
    pub quality_score: Dimensionless,
}

/// Physical or dimensionless value carried by a safety event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitoringMetric {
    /// Temperature rise above the baseline.
    TemperatureRise(TemperatureDifference),
    /// Mechanical index.
    MechanicalIndex(Dimensionless),
    /// Quality or resource metric.
    Dimensionless(Dimensionless),
}

/// Safety event log entry
#[derive(Debug, Clone)]
pub struct SafetyEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: MonitoringSafetyEventType,
    /// Parameter value with its physical meaning.
    pub parameter_value: MonitoringMetric,
    /// Safety limit with the same physical meaning as `parameter_value`.
    pub safety_limit: MonitoringMetric,
    /// Severity level
    pub severity: SafetySeverity,
    /// Human-readable description
    pub message: String,
}

/// Safety event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitoringSafetyEventType {
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

impl std::fmt::Display for MonitoringSafetyEventType {
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
pub struct MonitoringFrameMetrics {
    /// Total frames processed
    pub total_frames: usize,
    /// Frames with errors
    pub error_frames: usize,
    /// Average processing time.
    pub avg_processing_time: Time,
    /// Maximum processing time.
    pub max_processing_time: Time,
    /// Minimum processing time.
    pub min_processing_time: Time,
    /// Average frame rate.
    pub avg_frame_rate: Frequency,
    /// Uptime.
    pub uptime: Time,
}

/// Monitoring report
#[derive(Debug, Clone)]
pub struct MonitoringReport {
    /// Total uptime.
    pub uptime: Time,
    /// Total frames processed
    pub total_frames_processed: usize,
    /// Frames with errors
    pub error_frames: usize,
    /// Average frame rate.
    pub avg_frame_rate: Frequency,
    /// Average quality score (0-100).
    pub avg_quality_score: Dimensionless,
    /// Average processing time.
    pub avg_processing_time: Time,
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
