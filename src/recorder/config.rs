//! Recorder configuration

use crate::physics::sonoluminescence_detector::DetectorConfig;

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
}

impl RecorderConfig {
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
        }
    }

    pub fn with_pressure_recording(mut self, record: bool) -> Self {
        self.record_pressure = record;
        self
    }

    pub fn with_light_recording(mut self, record: bool) -> Self {
        self.record_light = record;
        self
    }

    pub fn with_temperature_recording(mut self, record: bool) -> Self {
        self.record_temperature = record;
        self
    }

    pub fn with_cavitation_detection(mut self, enable: bool, threshold: f64) -> Self {
        self.record_cavitation = enable;
        self.cavitation_threshold = threshold;
        self
    }

    pub fn with_sonoluminescence_detection(
        mut self,
        enable: bool,
        config: Option<DetectorConfig>,
    ) -> Self {
        self.record_sonoluminescence = enable;
        self.sl_detector_config = config;
        self
    }

    pub fn with_snapshot_interval(mut self, interval: usize) -> Self {
        self.snapshot_interval = interval;
        self
    }
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self::create("simulation_output")
    }
}