//! Recorder configuration

use crate::domain::sensor::sonoluminescence::DetectorConfig;

/// Recording mode for sensor data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordingMode {
    /// Record pressure at each time step (default)
    #[default]
    TimeSeries,
    /// Record maximum pressure (p_max)
    MaxPressure,
    /// Record minimum pressure (p_min)
    MinPressure,
    /// Record RMS pressure (p_rms)
    RmsPressure,
    /// Record final pressure (p_final)
    FinalPressure,
    /// Record maximum pressure over all time (p_max_all)
    MaxPressureAll,
    /// Record minimum pressure over all time (p_min_all)
    MinPressureAll,
    /// Record both max and min pressure
    MaxMinPressure,
    /// Record all statistics
    AllStatistics,
}

/// Configuration for recorder setup
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub filename: String,
    pub record_pressure: bool,
    pub record_light: bool,
    pub record_temperature: bool,
    pub record_cavitation: bool,
    pub record_sonoluminescence: bool,
    pub snapshot_interval: usize,
    /// Threshold for cavitation detection (Pa)
    pub cavitation_threshold: f64,
    /// Configuration for sonoluminescence detection
    pub sl_detector_config: Option<DetectorConfig>,
    /// Recording mode (k-Wave parity: p_max, p_min, p_rms, p_final)
    pub recording_mode: RecordingMode,
}

impl RecorderConfig {
    #[must_use]
    pub fn create(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            record_pressure: true,
            record_light: true,
            record_temperature: false,
            record_cavitation: false,
            record_sonoluminescence: false,
            snapshot_interval: 1,
            cavitation_threshold: -1e5, // -1 bar for cavitation
            sl_detector_config: None,
            recording_mode: RecordingMode::TimeSeries,
        }
    }

    #[must_use]
    pub fn with_pressure_recording(mut self, record: bool) -> Self {
        self.record_pressure = record;
        self
    }

    #[must_use]
    pub fn with_light_recording(mut self, record: bool) -> Self {
        self.record_light = record;
        self
    }

    #[must_use]
    pub fn with_temperature_recording(mut self, record: bool) -> Self {
        self.record_temperature = record;
        self
    }

    #[must_use]
    pub fn with_cavitation_detection(mut self, enable: bool, threshold: f64) -> Self {
        self.record_cavitation = enable;
        self.cavitation_threshold = threshold;
        self
    }

    #[must_use]
    pub fn with_sonoluminescence_detection(
        mut self,
        enable: bool,
        config: Option<DetectorConfig>,
    ) -> Self {
        self.record_sonoluminescence = enable;
        self.sl_detector_config = config;
        self
    }

    #[must_use]
    pub fn with_snapshot_interval(mut self, interval: usize) -> Self {
        self.snapshot_interval = interval;
        self
    }

    /// Set recording mode for k-Wave parity
    #[must_use]
    pub fn with_recording_mode(mut self, mode: RecordingMode) -> Self {
        self.recording_mode = mode;
        self
    }

    /// Enable maximum pressure recording (p_max)
    #[must_use]
    pub fn record_max_pressure(self) -> Self {
        self.with_recording_mode(RecordingMode::MaxPressure)
    }

    /// Enable minimum pressure recording (p_min)
    #[must_use]
    pub fn record_min_pressure(self) -> Self {
        self.with_recording_mode(RecordingMode::MinPressure)
    }

    /// Enable RMS pressure recording (p_rms)
    #[must_use]
    pub fn record_rms_pressure(self) -> Self {
        self.with_recording_mode(RecordingMode::RmsPressure)
    }

    /// Enable final pressure recording (p_final)
    #[must_use]
    pub fn record_final_pressure(self) -> Self {
        self.with_recording_mode(RecordingMode::FinalPressure)
    }
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self::create("simulation_output")
    }
}

impl RecordingMode {
    /// Returns true if this mode records time series data
    pub fn is_time_series(&self) -> bool {
        matches!(self, RecordingMode::TimeSeries)
    }

    /// Returns true if this mode records statistics (max, min, rms, final)
    pub fn is_statistical(&self) -> bool {
        !matches!(self, RecordingMode::TimeSeries)
    }

    /// Get the corresponding field name in k-Wave output
    pub fn kwave_field_name(&self) -> &'static str {
        match self {
            RecordingMode::TimeSeries => "p",
            RecordingMode::MaxPressure => "p_max",
            RecordingMode::MinPressure => "p_min",
            RecordingMode::RmsPressure => "p_rms",
            RecordingMode::FinalPressure => "p_final",
            RecordingMode::MaxPressureAll => "p_max_all",
            RecordingMode::MinPressureAll => "p_min_all",
            RecordingMode::MaxMinPressure => "p_max_min",
            RecordingMode::AllStatistics => "p_all_stats",
        }
    }
}
