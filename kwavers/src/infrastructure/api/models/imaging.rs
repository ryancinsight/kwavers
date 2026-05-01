//! Real-time ultrasound frame and imaging parameters.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct UltrasoundFrame {
    /// Frame sequence number
    pub frame_id: u64,
    /// Device identifier
    pub device_id: String,
    /// Timestamp when frame was captured
    pub timestamp: DateTime<Utc>,
    /// RF data dimensions [time_samples, channels, spatial_points]
    pub dimensions: Vec<usize>,
    /// RF data as base64-encoded bytes
    pub rf_data: String,
    /// Imaging parameters
    pub parameters: ImagingParameters,
    /// Device metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Imaging parameters for beamforming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagingParameters {
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Center frequency (Hz)
    pub center_frequency: f64,
    /// Number of active elements
    pub num_elements: usize,
    /// Element spacing (m)
    pub element_spacing: f64,
    /// Steering angles for each frame (radians)
    pub steering_angles: Vec<f64>,
    /// Depth range [start, end] in meters
    pub depth_range: [f64; 2],
}
