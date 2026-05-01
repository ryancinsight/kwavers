//! Device connectivity and ultrasound hardware abstractions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub enum DeviceType {
    /// Standard ultrasound system
    Ultrasound,
    /// Handheld point-of-care device
    Handheld,
    /// Robotic ultrasound probe
    Robotic,
    /// Simulated device for testing
    Simulated,
}

/// Device capabilities for clinical workflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCapability {
    /// 2D B-mode imaging
    Imaging2D,
    /// 3D/4D volumetric imaging
    Imaging3D,
    /// Doppler flow analysis
    Doppler,
    /// Color flow mapping
    ColorFlow,
    /// Elastography tissue characterization
    Elastography,
    /// Contrast-enhanced ultrasound
    ContrastEnhanced,
}

/// Comprehensive ultrasound device information for point-of-care integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub id: String,
    /// Type of ultrasound device
    pub device_type: DeviceType,
    /// Device model name
    pub model: String,
    /// Device manufacturer
    pub manufacturer: String,
    /// List of supported clinical capabilities
    pub capabilities: Vec<DeviceCapability>,
    /// Current operational status
    pub status: DeviceStatus,
    /// Timestamp of last calibration
    pub last_calibration: DateTime<Utc>,
    /// Firmware version string
    pub firmware_version: String,
}

/// Ultrasound device information for point-of-care integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrasoundDevice {
    /// Unique device identifier
    pub device_id: String,
    /// Device model/manufacturer
    pub model: String,
    /// Device capabilities (linear, convex, phased array, etc.)
    pub capabilities: Vec<String>,
    /// Supported imaging modes
    pub imaging_modes: Vec<String>,
    /// Maximum frame rate (Hz)
    pub max_frame_rate: u32,
    /// Battery level (0-100)
    pub battery_level: Option<u8>,
    /// Device status
    pub status: DeviceStatus,
    /// Last seen timestamp
    pub last_seen: DateTime<Utc>,
}

/// Device connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceStatus {
    Connected,
    Disconnected,
    Error,
    Charging,
    Available,
    InUse,
    Calibrating,
}

/// Real-time ultrasound frame data for AI processing
