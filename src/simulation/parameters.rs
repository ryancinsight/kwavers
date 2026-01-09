//! Simulation parameters configuration

use serde::{Deserialize, Serialize};

/// Core simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    /// Simulation duration in seconds
    pub duration: f64,
    /// Time step (auto-calculated if None)
    pub dt: Option<f64>,
    /// CFL number for stability
    pub cfl: f64,
    /// Reference frequency in Hz
    pub frequency: f64,
    /// Enable nonlinear effects
    pub nonlinear: bool,
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Simulation type
    pub simulation_type: SimulationType,
}

/// Types of simulations supported
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SimulationType {
    /// Forward wave propagation
    Forward,
    /// Time reversal reconstruction
    TimeReversal,
    /// Photoacoustic imaging
    Photoacoustic,
    /// Full waveform inversion
    FullWaveformInversion,
    /// Therapy planning
    Therapy,
}

impl SimulationParameters {
    /// Validate simulation parameters
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        if self.duration <= 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "duration".to_string(),
                value: self.duration.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if let Some(dt) = self.dt {
            if dt <= 0.0 {
                return Err(crate::core::error::ConfigError::InvalidValue {
                    parameter: "dt".to_string(),
                    value: dt.to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "cfl".to_string(),
                value: self.cfl.to_string(),
                constraint: "Must be in (0, 1]".to_string(),
            }
            .into());
        }

        if self.frequency <= 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.temperature < 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "temperature".to_string(),
                value: self.temperature.to_string(),
                constraint: "Must be non-negative (Kelvin)".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            duration: 1e-3,
            dt: None,
            cfl: 0.3,
            frequency: 1e6,
            nonlinear: false,
            temperature: 293.15, // 20°C
            simulation_type: SimulationType::Forward,
        }
    }
}
