//! Recorder configuration

use crate::sonoluminescence::DetectorConfig;

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

/// Whether a recorder channel is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordingState {
    /// Do not collect this channel.
    #[default]
    Disabled,
    /// Collect this channel.
    Enabled,
}

/// Recorder channels that are active for a simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecordingChannels(u8);

/// A recorder channel selected in [`RecordingChannels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecorderChannel {
    /// Acoustic pressure samples and statistics.
    Pressure,
    /// Optical light samples and statistics.
    Light,
    /// Temperature samples and thermal events.
    Temperature,
    /// Cavitation event detection.
    Cavitation,
    /// Sonoluminescence event detection.
    Sonoluminescence,
}

impl RecordingChannels {
    const PRESSURE: u8 = 1 << 0;
    const LIGHT: u8 = 1 << 1;
    const TEMPERATURE: u8 = 1 << 2;
    const CAVITATION: u8 = 1 << 3;
    const SONOLUMINESCENCE: u8 = 1 << 4;

    /// Return an empty channel selection.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Return a selection with `channel` set to `state`.
    #[must_use]
    pub const fn with(self, channel: RecorderChannel, state: RecordingState) -> Self {
        let mask = match channel {
            RecorderChannel::Pressure => Self::PRESSURE,
            RecorderChannel::Light => Self::LIGHT,
            RecorderChannel::Temperature => Self::TEMPERATURE,
            RecorderChannel::Cavitation => Self::CAVITATION,
            RecorderChannel::Sonoluminescence => Self::SONOLUMINESCENCE,
        };
        match state {
            RecordingState::Disabled => Self(self.0 & !mask),
            RecordingState::Enabled => Self(self.0 | mask),
        }
    }

    /// Report whether `channel` is enabled.
    #[must_use]
    pub const fn contains(self, channel: RecorderChannel) -> bool {
        let mask = match channel {
            RecorderChannel::Pressure => Self::PRESSURE,
            RecorderChannel::Light => Self::LIGHT,
            RecorderChannel::Temperature => Self::TEMPERATURE,
            RecorderChannel::Cavitation => Self::CAVITATION,
            RecorderChannel::Sonoluminescence => Self::SONOLUMINESCENCE,
        };
        self.0 & mask != 0
    }
}

/// Configuration for recorder setup
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub filename: String,
    /// Channels collected by the recorder.
    pub channels: RecordingChannels,
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
            filename: filename.to_owned(),
            channels: RecordingChannels::empty()
                .with(RecorderChannel::Pressure, RecordingState::Enabled)
                .with(RecorderChannel::Light, RecordingState::Enabled),
            snapshot_interval: 1,
            cavitation_threshold: -1e5, // -1 bar for cavitation
            sl_detector_config: None,
            recording_mode: RecordingMode::TimeSeries,
        }
    }

    #[must_use]
    pub fn with_pressure_recording(mut self, state: RecordingState) -> Self {
        self.channels = self.channels.with(RecorderChannel::Pressure, state);
        self
    }

    #[must_use]
    pub fn with_light_recording(mut self, state: RecordingState) -> Self {
        self.channels = self.channels.with(RecorderChannel::Light, state);
        self
    }

    #[must_use]
    pub fn with_temperature_recording(mut self, state: RecordingState) -> Self {
        self.channels = self.channels.with(RecorderChannel::Temperature, state);
        self
    }

    #[must_use]
    pub fn with_cavitation_detection(mut self, state: RecordingState, threshold: f64) -> Self {
        self.channels = self.channels.with(RecorderChannel::Cavitation, state);
        self.cavitation_threshold = threshold;
        self
    }

    #[must_use]
    pub fn with_sonoluminescence_detection(
        mut self,
        state: RecordingState,
        config: Option<DetectorConfig>,
    ) -> Self {
        self.channels = self.channels.with(RecorderChannel::Sonoluminescence, state);
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
    #[must_use]
    pub fn is_time_series(&self) -> bool {
        matches!(self, Self::TimeSeries)
    }

    /// Returns true if this mode records statistics (max, min, rms, final)
    #[must_use]
    pub fn is_statistical(&self) -> bool {
        !matches!(self, Self::TimeSeries)
    }

    /// Get the corresponding field name in k-Wave output
    #[must_use]
    pub fn kwave_field_name(&self) -> &'static str {
        match self {
            Self::TimeSeries => "p",
            Self::MaxPressure => "p_max",
            Self::MinPressure => "p_min",
            Self::RmsPressure => "p_rms",
            Self::FinalPressure => "p_final",
            Self::MaxPressureAll => "p_max_all",
            Self::MinPressureAll => "p_min_all",
            Self::MaxMinPressure => "p_max_min",
            Self::AllStatistics => "p_all_stats",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RecorderChannel, RecorderConfig, RecordingState};

    #[test]
    fn channel_selection_preserves_independent_states() {
        let config = RecorderConfig::create("output")
            .with_pressure_recording(RecordingState::Disabled)
            .with_temperature_recording(RecordingState::Enabled)
            .with_cavitation_detection(RecordingState::Enabled, -2.0e5)
            .with_sonoluminescence_detection(RecordingState::Disabled, None);

        assert!(!config.channels.contains(RecorderChannel::Pressure));
        assert!(config.channels.contains(RecorderChannel::Light));
        assert!(config.channels.contains(RecorderChannel::Temperature));
        assert!(config.channels.contains(RecorderChannel::Cavitation));
        assert!(!config.channels.contains(RecorderChannel::Sonoluminescence));
        assert_eq!(config.cavitation_threshold, -2.0e5);
    }
}
