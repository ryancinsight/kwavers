//! Control strategies for cavitation feedback

use serde::{Deserialize, Serialize};

/// Control strategies for feedback
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ControlStrategy {
    /// Control only amplitude
    AmplitudeOnly,
    /// Control only duty cycle  
    DutyCycleOnly,
    /// Control both amplitude and duty cycle
    Combined,
    /// Cascaded control (coarse/fine)
    Cascaded,
    /// Model predictive control
    Predictive,
}

impl ControlStrategy {
    /// Determine if amplitude control is enabled
    pub fn uses_amplitude(&self) -> bool {
        matches!(self, Self::AmplitudeOnly | Self::Combined | Self::Cascaded)
    }
    
    /// Determine if duty cycle control is enabled
    pub fn uses_duty_cycle(&self) -> bool {
        matches!(self, Self::DutyCycleOnly | Self::Combined | Self::Cascaded)
    }
    
    /// Determine if predictive control is enabled
    pub fn is_predictive(&self) -> bool {
        matches!(self, Self::Predictive)
    }
}

impl Default for ControlStrategy {
    fn default() -> Self {
        Self::Combined
    }
}
