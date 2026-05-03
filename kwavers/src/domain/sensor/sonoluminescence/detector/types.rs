//! Data types for sonoluminescence detection.

use super::constants;

/// Sonoluminescence event data
#[derive(Debug, Clone)]
pub struct SonoluminescenceEvent {
    /// Time of event (s)
    pub time: f64,
    /// Position indices (i, j, k)
    pub position: (usize, usize, usize),
    /// Physical position (x, y, z) in meters
    pub physical_position: (f64, f64, f64),
    /// Peak temperature (K)
    pub peak_temperature: f64,
    /// Peak pressure (Pa)
    pub peak_pressure: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Total photon count
    pub photon_count: f64,
    /// Peak wavelength (m)
    pub peak_wavelength: f64,
    /// Event duration (s)
    pub duration: f64,
    /// Energy released (J)
    pub energy: f64,
}

/// Sonoluminescence detector configuration
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Enable spectral analysis
    pub spectral_analysis: bool,
    /// Enable time-resolved detection
    pub time_resolved: bool,
    /// Temperature threshold (K)
    pub temperature_threshold: f64,
    /// Pressure threshold (Pa)
    pub pressure_threshold: f64,
    /// Compression ratio threshold
    pub compression_threshold: f64,
    /// Spatial resolution for clustering (m)
    pub spatial_resolution: f64,
    /// Time resolution for clustering (s)
    pub time_resolution: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            spectral_analysis: true,
            time_resolved: true,
            temperature_threshold: constants::MIN_TEMPERATURE_SL,
            pressure_threshold: constants::MIN_PRESSURE_SL,
            compression_threshold: constants::MAX_COMPRESSION_RATIO,
            spatial_resolution: 1e-6, // 1 μm
            time_resolution: 1e-10,   // 100 ps
        }
    }
}

/// Statistics about sonoluminescence events
#[derive(Debug, Default, Clone)]
pub struct SonoluminescenceStatistics {
    /// Total number of events detected
    pub total_events: usize,
    /// Total photon count
    pub total_photons: f64,
    /// Total energy released (J)
    pub total_energy: f64,
    /// Maximum temperature observed (K)
    pub max_temperature: f64,
    /// Average temperature of events (K)
    pub avg_temperature: f64,
    /// Event rate (events/s)
    pub event_rate: f64,
}
