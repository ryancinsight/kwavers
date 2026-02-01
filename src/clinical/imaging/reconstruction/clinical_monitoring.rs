//! Clinical Real-Time Reconstruction Monitoring
//!
//! This module provides comprehensive monitoring and quality assurance for
//! real-time ultrasound reconstruction pipelines in clinical deployment.
//!
//! ## Monitoring Architecture
//!
//! ```
//! Reconstruction Pipeline
//!     ↓
//! Frame Capture → Quality Assessment → Safety Checks → Storage
//!     ↓              ↓                   ↓              ↓
//! Timestamp    SNR, Resolution    MI, Dose        Archive
//! Metadata     Artifacts          Temperature      Audit Log
//! ```
//!
//! ## Key Metrics
//!
//! - **Image Quality**: SNR, contrast, resolution
//! - **Safety Parameters**: Mechanical index, thermal dose, temperature
//! - **Performance**: Frame rate, latency, throughput
//! - **System Health**: CPU usage, memory, error rates
//!
//! ## Real-Time Constraints
//!
//! - Frame processing: < 100ms per frame (10 fps minimum)
//! - Monitoring overhead: < 10% of processing time
//! - Audit logging: < 5ms per event
//!
//! ## References
//! - FDA (2010): "Ultrasound Quality Standards"
//! - IEC 60601-2-37: Therapeutic ultrasound safety and quality
//! - AIUM (2014): "Quality Standards for Real-Time Ultrasound Equipment"

use crate::clinical::safety::{ComplianceStatus, EnhancedComplianceCheck};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

/// Real-time clinical monitoring system
#[derive(Debug)]
pub struct ClinicalMonitor {
    /// Configuration for monitoring
    config: MonitoringConfig,
    /// Frame quality history
    frame_history: VecDeque<FrameQualityRecord>,
    /// Safety event log
    safety_log: VecDeque<SafetyEvent>,
    /// System performance metrics
    performance_metrics: PerformanceMetrics,
    /// Monitoring start time
    start_time: Instant,
}

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
            quality_alert_threshold: 0.7, // Alert if quality < 70% of baseline
            safety_alert_threshold: 0.9,  // Alert if safety metric > 90% of limit
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

impl ClinicalMonitor {
    /// Create new clinical monitor
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            frame_history: VecDeque::with_capacity(config.history_window),
            safety_log: VecDeque::with_capacity(1000),
            performance_metrics: PerformanceMetrics::default(),
            start_time: Instant::now(),
        }
    }

    /// Record frame quality metrics
    pub fn record_frame_quality(
        &mut self,
        frame_number: usize,
        processing_time_ms: f64,
        snr_db: f64,
        contrast: f64,
        spatial_resolution_mm: f64,
        artifact_level: f64,
    ) -> KwaversResult<()> {
        // Compute quality score (0-100)
        let quality_score = self.compute_quality_score(snr_db, contrast, artifact_level);

        let record = FrameQualityRecord {
            frame_number,
            timestamp: SystemTime::now(),
            processing_time_ms,
            snr_db,
            contrast,
            spatial_resolution_mm,
            artifact_level,
            quality_score,
        };

        // Update performance metrics
        self.performance_metrics.total_frames += 1;
        self.performance_metrics.avg_processing_time_ms =
            (self.performance_metrics.avg_processing_time_ms
                * (self.performance_metrics.total_frames - 1) as f64
                + processing_time_ms)
                / self.performance_metrics.total_frames as f64;
        self.performance_metrics.max_processing_time_ms = self
            .performance_metrics
            .max_processing_time_ms
            .max(processing_time_ms);
        self.performance_metrics.min_processing_time_ms =
            if self.performance_metrics.total_frames == 1 {
                processing_time_ms
            } else {
                self.performance_metrics
                    .min_processing_time_ms
                    .min(processing_time_ms)
            };

        // Check for quality alerts
        if quality_score < self.config.quality_alert_threshold * 100.0 {
            self.log_safety_event(SafetyEvent {
                timestamp: SystemTime::now(),
                event_type: SafetyEventType::QualityDegradation,
                parameter_value: quality_score,
                safety_limit: self.config.quality_alert_threshold * 100.0,
                severity: SafetySeverity::Warning,
                message: format!("Frame quality score {:.1} below threshold", quality_score),
            })?;
        }

        // Add to history
        self.frame_history.push_back(record);
        if self.frame_history.len() > self.config.history_window {
            self.frame_history.pop_front();
        }

        Ok(())
    }

    /// Record safety event
    pub fn log_safety_event(&mut self, event: SafetyEvent) -> KwaversResult<()> {
        self.safety_log.push_back(event.clone());

        // Keep log size bounded
        if self.safety_log.len() > 10000 {
            self.safety_log.pop_front();
        }

        Ok(())
    }

    /// Check temperature safety
    pub fn check_temperature(&mut self, current_temperature_c: f64, baseline_temp_c: f64) {
        let temp_rise = current_temperature_c - baseline_temp_c;

        if temp_rise > self.config.max_temperature_rise_c {
            let _ = self.log_safety_event(SafetyEvent {
                timestamp: SystemTime::now(),
                event_type: SafetyEventType::TemperatureExceeded,
                parameter_value: temp_rise,
                safety_limit: self.config.max_temperature_rise_c,
                severity: SafetySeverity::Critical,
                message: format!(
                    "Temperature rise {:.1}°C exceeds limit {:.1}°C",
                    temp_rise, self.config.max_temperature_rise_c
                ),
            });
        } else if temp_rise > self.config.max_temperature_rise_c * 0.8 {
            let _ = self.log_safety_event(SafetyEvent {
                timestamp: SystemTime::now(),
                event_type: SafetyEventType::TemperatureExceeded,
                parameter_value: temp_rise,
                safety_limit: self.config.max_temperature_rise_c,
                severity: SafetySeverity::Urgent,
                message: format!(
                    "Temperature rise {:.1}°C approaching limit {:.1}°C",
                    temp_rise, self.config.max_temperature_rise_c
                ),
            });
        }
    }

    /// Check mechanical index safety
    pub fn check_mechanical_index(&mut self, mechanical_index: f64) {
        if mechanical_index > self.config.max_mechanical_index {
            let _ = self.log_safety_event(SafetyEvent {
                timestamp: SystemTime::now(),
                event_type: SafetyEventType::MechanicalIndexExceeded,
                parameter_value: mechanical_index,
                safety_limit: self.config.max_mechanical_index,
                severity: SafetySeverity::Critical,
                message: format!(
                    "MI {:.2} exceeds safety limit {:.2}",
                    mechanical_index, self.config.max_mechanical_index
                ),
            });
        } else if mechanical_index > self.config.max_mechanical_index * 0.8 {
            let _ = self.log_safety_event(SafetyEvent {
                timestamp: SystemTime::now(),
                event_type: SafetyEventType::MechanicalIndexExceeded,
                parameter_value: mechanical_index,
                safety_limit: self.config.max_mechanical_index,
                severity: SafetySeverity::Urgent,
                message: format!(
                    "MI {:.2} approaching safety limit {:.2}",
                    mechanical_index, self.config.max_mechanical_index
                ),
            });
        }
    }

    /// Compute quality score (0-100) from metrics
    fn compute_quality_score(&self, snr_db: f64, contrast: f64, artifact_level: f64) -> f64 {
        // Weighted quality score
        // SNR: 0 dB → 0%, 30 dB → 100%
        let snr_score = (snr_db / 30.0 * 100.0).min(100.0).max(0.0);

        // Contrast: 0 → 0%, 1.0 → 100%
        let contrast_score = (contrast * 100.0).min(100.0).max(0.0);

        // Artifact level: 0 → 100%, 1.0 → 0%
        let artifact_score = ((1.0 - artifact_level) * 100.0).min(100.0).max(0.0);

        // Weighted average: SNR=40%, Contrast=40%, Artifacts=20%
        (snr_score * 0.4 + contrast_score * 0.4 + artifact_score * 0.2).round()
    }

    /// Get frame quality history
    pub fn frame_history(&self) -> Vec<FrameQualityRecord> {
        self.frame_history.iter().cloned().collect()
    }

    /// Get safety event log
    pub fn safety_log(&self) -> Vec<SafetyEvent> {
        self.safety_log.iter().cloned().collect()
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Get session summary report
    pub fn generate_report(&self) -> MonitoringReport {
        let uptime = self.start_time.elapsed().as_secs_f64();

        // Calculate average quality score
        let avg_quality = if !self.frame_history.is_empty() {
            self.frame_history
                .iter()
                .map(|r| r.quality_score)
                .sum::<f64>()
                / self.frame_history.len() as f64
        } else {
            0.0
        };

        // Count safety events by severity
        let info_count = self
            .safety_log
            .iter()
            .filter(|e| e.severity == SafetySeverity::Info)
            .count();
        let warning_count = self
            .safety_log
            .iter()
            .filter(|e| e.severity == SafetySeverity::Warning)
            .count();
        let urgent_count = self
            .safety_log
            .iter()
            .filter(|e| e.severity == SafetySeverity::Urgent)
            .count();
        let critical_count = self
            .safety_log
            .iter()
            .filter(|e| e.severity == SafetySeverity::Critical)
            .count();

        MonitoringReport {
            uptime_seconds: uptime,
            total_frames_processed: self.performance_metrics.total_frames,
            error_frames: self.performance_metrics.error_frames,
            avg_frame_rate_fps: self.performance_metrics.total_frames as f64 / uptime.max(1.0),
            avg_quality_score: avg_quality,
            avg_processing_time_ms: self.performance_metrics.avg_processing_time_ms,
            info_events: info_count,
            warning_events: warning_count,
            urgent_events: urgent_count,
            critical_events: critical_count,
            system_status: if critical_count > 0 {
                "UNSAFE - Critical events detected".to_string()
            } else if urgent_count > 0 {
                "CAUTION - Urgent events detected".to_string()
            } else {
                "SAFE - Normal operation".to_string()
            },
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert!(config.enable_quality_monitoring);
        assert!(config.enable_safety_logging);
    }

    #[test]
    fn test_clinical_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = ClinicalMonitor::new(config);
        assert_eq!(monitor.performance_metrics.total_frames, 0);
    }

    #[test]
    fn test_frame_quality_recording() {
        let config = MonitoringConfig::default();
        let mut monitor = ClinicalMonitor::new(config);

        let result = monitor.record_frame_quality(1, 50.0, 25.0, 0.8, 1.5, 0.1);
        assert!(result.is_ok());
        assert_eq!(monitor.performance_metrics.total_frames, 1);
    }

    #[test]
    fn test_safety_event_logging() {
        let config = MonitoringConfig::default();
        let mut monitor = ClinicalMonitor::new(config);

        let event = SafetyEvent {
            timestamp: SystemTime::now(),
            event_type: SafetyEventType::TemperatureExceeded,
            parameter_value: 6.0,
            safety_limit: 5.0,
            severity: SafetySeverity::Critical,
            message: "Temperature exceeds limit".to_string(),
        };

        let result = monitor.log_safety_event(event);
        assert!(result.is_ok());
        assert_eq!(monitor.safety_log().len(), 1);
    }

    #[test]
    fn test_temperature_check() {
        let config = MonitoringConfig::default();
        let mut monitor = ClinicalMonitor::new(config);

        monitor.check_temperature(42.5, 37.0); // 5.5°C rise
        assert_eq!(monitor.safety_log().len(), 1);

        let event = &monitor.safety_log()[0];
        assert_eq!(event.severity, SafetySeverity::Critical);
    }

    #[test]
    fn test_mechanical_index_check() {
        let config = MonitoringConfig::default();
        let mut monitor = ClinicalMonitor::new(config);

        monitor.check_mechanical_index(2.0); // Exceeds limit of 1.9
        assert_eq!(monitor.safety_log().len(), 1);
    }

    #[test]
    fn test_monitoring_report() {
        let config = MonitoringConfig::default();
        let monitor = ClinicalMonitor::new(config);

        let report = monitor.generate_report();
        assert_eq!(report.total_frames_processed, 0);
        assert!(report.uptime_seconds >= 0.0);
    }

    #[test]
    fn test_quality_score_computation() {
        let config = MonitoringConfig::default();
        let monitor = ClinicalMonitor::new(config);

        // Perfect quality
        let score = monitor.compute_quality_score(30.0, 1.0, 0.0);
        assert!(score > 90.0);

        // Poor quality
        let score = monitor.compute_quality_score(5.0, 0.2, 0.8);
        assert!(score < 30.0);
    }

    #[test]
    fn test_safety_event_severity() {
        assert!(SafetySeverity::Critical > SafetySeverity::Warning);
        assert!(SafetySeverity::Urgent > SafetySeverity::Info);
    }
}
