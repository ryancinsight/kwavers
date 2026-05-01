//! Mobile-optimized workflow and power/network constraint types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct MobileOptimizationRequest {
    /// Device capabilities
    pub device_capabilities: DeviceCapabilities,
    /// Network conditions
    pub network_conditions: NetworkConditions,
    /// Power management settings
    pub power_settings: PowerSettings,
    /// Target performance requirements
    pub performance_targets: PerformanceTargets,
}

/// Device capabilities for mobile optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// CPU cores available
    pub cpu_cores: usize,
    /// RAM available (MB)
    pub ram_mb: usize,
    /// GPU available
    pub has_gpu: bool,
    /// SIMD support
    pub has_simd: bool,
    /// Battery capacity (mAh)
    pub battery_mah: Option<u32>,
}

/// Network conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Connection type
    pub connection_type: ConnectionType,
    /// Bandwidth (Mbps)
    pub bandwidth_mbps: f64,
    /// Latency (ms)
    pub latency_ms: u64,
    /// Packet loss (%)
    pub packet_loss_percent: f64,
}

/// Connection types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionType {
    Wifi,
    Cellular4G,
    Cellular5G,
    Ethernet,
    Bluetooth,
}

/// Power management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSettings {
    /// Battery level (0-100)
    pub battery_level: u8,
    /// Power saving mode enabled
    pub power_saving_mode: bool,
    /// Screen brightness (0-100)
    pub screen_brightness: u8,
    /// Thermal throttling active
    pub thermal_throttling: bool,
}

/// Performance targets for mobile devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target frame rate (Hz)
    pub target_frame_rate_hz: f64,
    /// Maximum latency (ms)
    pub max_latency_ms: u64,
    /// Acceptable image quality (0-1)
    pub acceptable_quality: f32,
    /// Battery usage limit (% per hour)
    pub battery_usage_limit_percent: f64,
}

/// Mobile optimization response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileOptimizationResponse {
    /// Recommended processing configuration
    pub recommended_config: ProcessingConfig,
    /// Performance predictions
    pub performance_predictions: PerformancePredictions,
    /// Power consumption estimates
    pub power_estimates: PowerEstimates,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

/// Processing configuration for mobile devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Frame rate to use (Hz)
    pub frame_rate_hz: f64,
    /// Image resolution scaling factor
    pub resolution_scale: f64,
    /// AI model precision (fp32, fp16, int8)
    pub model_precision: String,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// Batch processing size
    pub batch_size: usize,
}

/// Performance predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictions {
    /// Predicted latency (ms)
    pub predicted_latency_ms: f64,
    /// Predicted frame rate (Hz)
    pub predicted_frame_rate_hz: f64,
    /// Predicted image quality (0-1)
    pub predicted_quality: f32,
    /// Confidence in predictions (0-1)
    pub prediction_confidence: f32,
}

/// Power consumption estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEstimates {
    /// CPU usage (%)
    pub cpu_usage_percent: f64,
    /// GPU usage (%)
    pub gpu_usage_percent: Option<f64>,
    /// Battery drain rate (% per hour)
    pub battery_drain_percent_per_hour: f64,
    /// Thermal impact score (0-1)
    pub thermal_impact: f64,
}
