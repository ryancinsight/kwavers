use super::types::{
    ClinicalMonitoringConfig, FrameQualityRecord, MonitoringFrameMetrics, MonitoringReport,
    MonitoringSafetyEventType, SafetyEvent, SafetySeverity,
};
use kwavers_core::error::KwaversResult;
use std::collections::VecDeque;
use std::time::{Instant, SystemTime};

/// Real-time clinical monitoring system
#[derive(Debug)]
pub struct ClinicalMonitor {
    /// Configuration for monitoring
    pub(super) config: ClinicalMonitoringConfig,
    /// Frame quality history
    frame_history: VecDeque<FrameQualityRecord>,
    /// Safety event log
    safety_log: VecDeque<SafetyEvent>,
    /// System performance metrics
    pub(super) performance_metrics: MonitoringFrameMetrics,
    /// Monitoring start time
    start_time: Instant,
}

impl ClinicalMonitor {
    /// Create new clinical monitor
    #[must_use]
    pub fn new(config: ClinicalMonitoringConfig) -> Self {
        let history_cap = config.history_window;
        Self {
            config,
            frame_history: VecDeque::with_capacity(history_cap),
            safety_log: VecDeque::with_capacity(1000),
            performance_metrics: MonitoringFrameMetrics::default(),
            start_time: Instant::now(),
        }
    }

    /// Record frame quality metrics
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn record_frame_quality(
        &mut self,
        frame_number: usize,
        processing_time_ms: f64,
        snr_db: f64,
        contrast: f64,
        spatial_resolution_mm: f64,
        artifact_level: f64,
    ) -> KwaversResult<()> {
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

        self.performance_metrics.total_frames += 1;
        self.performance_metrics.avg_processing_time_ms =
            self.performance_metrics.avg_processing_time_ms.mul_add(
                (self.performance_metrics.total_frames - 1) as f64,
                processing_time_ms,
            ) / self.performance_metrics.total_frames as f64;
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

        if quality_score < self.config.quality_alert_threshold * 100.0 {
            self.log_safety_event(SafetyEvent {
                timestamp: SystemTime::now(),
                event_type: MonitoringSafetyEventType::QualityDegradation,
                parameter_value: quality_score,
                safety_limit: self.config.quality_alert_threshold * 100.0,
                severity: SafetySeverity::Warning,
                message: format!("Frame quality score {:.1} below threshold", quality_score),
            })?;
        }

        self.frame_history.push_back(record);
        if self.frame_history.len() > self.config.history_window {
            self.frame_history.pop_front();
        }

        Ok(())
    }

    /// Record safety event
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn log_safety_event(&mut self, event: SafetyEvent) -> KwaversResult<()> {
        self.safety_log.push_back(event);

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
                event_type: MonitoringSafetyEventType::TemperatureExceeded,
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
                event_type: MonitoringSafetyEventType::TemperatureExceeded,
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
                event_type: MonitoringSafetyEventType::MechanicalIndexExceeded,
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
                event_type: MonitoringSafetyEventType::MechanicalIndexExceeded,
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
    ///
    /// - SNR: 0 dB → 0%, 30 dB → 100% (weight 40%)
    /// - Contrast: 0 → 0%, 1.0 → 100% (weight 40%)
    /// - Artifact: 0 → 100%, 1.0 → 0% (weight 20%)
    pub(super) fn compute_quality_score(
        &self,
        snr_db: f64,
        contrast: f64,
        artifact_level: f64,
    ) -> f64 {
        let snr_score = (snr_db / 30.0 * 100.0).clamp(0.0, 100.0);
        let contrast_score = (contrast * 100.0).clamp(0.0, 100.0);
        let artifact_score = ((1.0 - artifact_level) * 100.0).clamp(0.0, 100.0);
        (snr_score * 0.4 + contrast_score * 0.4 + artifact_score * 0.2).round()
    }

    /// Get frame quality history
    #[must_use]
    pub fn frame_history(&self) -> Vec<FrameQualityRecord> {
        self.frame_history.iter().cloned().collect()
    }

    /// Get safety event log
    #[must_use]
    pub fn safety_log(&self) -> Vec<SafetyEvent> {
        self.safety_log.iter().cloned().collect()
    }

    /// Get performance metrics
    #[must_use]
    pub fn performance_metrics(&self) -> &MonitoringFrameMetrics {
        &self.performance_metrics
    }

    /// Get session summary report
    #[must_use]
    pub fn generate_report(&self) -> MonitoringReport {
        let uptime = self.start_time.elapsed().as_secs_f64();

        let avg_quality = if !self.frame_history.is_empty() {
            self.frame_history
                .iter()
                .map(|r| r.quality_score)
                .sum::<f64>()
                / self.frame_history.len() as f64
        } else {
            0.0
        };

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
                "UNSAFE - Critical events detected".to_owned()
            } else if urgent_count > 0 {
                "CAUTION - Urgent events detected".to_owned()
            } else {
                "SAFE - Normal operation".to_owned()
            },
        }
    }
}
