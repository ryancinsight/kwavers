//! Configuration for feedback control system

use super::strategy::ControlStrategy;
use crate::physics::cavitation_control::pid_controller::PIDGains;
use serde::{Deserialize, Serialize};

/// Default target cavitation intensity (0-1)
pub const DEFAULT_TARGET_INTENSITY: f64 = 0.5;

/// Control loop update rate (Hz)
pub const CONTROL_UPDATE_RATE: f64 = 100.0;

/// Minimum control update period (seconds)
pub const MIN_UPDATE_PERIOD: f64 = 0.001;

/// Control dead zone to prevent oscillations
pub const CONTROL_DEAD_ZONE: f64 = 0.02;

/// Safety shutdown threshold
pub const SAFETY_SHUTDOWN_THRESHOLD: f64 = 0.95;

/// Feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Target cavitation intensity (0-1)
    pub target_intensity: f64,
    /// Control strategy to use
    pub strategy: ControlStrategy,
    /// PID gains for amplitude control
    pub amplitude_gains: PIDGains,
    /// PID gains for duty cycle control
    pub duty_cycle_gains: PIDGains,
    /// Control update rate (Hz)
    pub update_rate: f64,
    /// Enable safety monitoring
    pub enable_safety: bool,
    /// Maximum allowed intensity
    pub max_intensity: f64,
    /// Control dead zone
    pub dead_zone: f64,
}

impl FeedbackConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.target_intensity < 0.0 || self.target_intensity > 1.0 {
            return Err(format!(
                "Target intensity must be in [0, 1], got {}",
                self.target_intensity
            ));
        }
        
        if self.update_rate <= 0.0 {
            return Err(format!(
                "Update rate must be positive, got {}",
                self.update_rate
            ));
        }
        
        if self.max_intensity <= self.target_intensity {
            return Err(format!(
                "Max intensity {} must exceed target {}",
                self.max_intensity, self.target_intensity
            ));
        }
        
        if self.dead_zone < 0.0 || self.dead_zone > 0.1 {
            return Err(format!(
                "Dead zone must be in [0, 0.1], got {}",
                self.dead_zone
            ));
        }
        
        Ok(())
    }
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            target_intensity: DEFAULT_TARGET_INTENSITY,
            strategy: ControlStrategy::default(),
            amplitude_gains: PIDGains::default(),
            duty_cycle_gains: PIDGains::default(),
            update_rate: CONTROL_UPDATE_RATE,
            enable_safety: true,
            max_intensity: 0.9,
            dead_zone: CONTROL_DEAD_ZONE,
        }
    }
}
