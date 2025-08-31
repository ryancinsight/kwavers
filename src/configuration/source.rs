//! Source configuration parameters

use serde::{Deserialize, Serialize};

/// Source configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceParameters {
    /// Source type
    pub source_type: SourceType,
    /// Source amplitude in Pa
    pub amplitude: f64,
    /// Source frequency in Hz
    pub frequency: f64,
    /// Source position [x, y, z] in meters
    pub position: [f64; 3],
    /// Source size/radius in meters
    pub radius: f64,
    /// Time delay in seconds
    pub delay: f64,
    /// Pulse parameters
    pub pulse: PulseParameters,
}

/// Types of acoustic sources
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    /// Point source
    Point,
    /// Focused transducer
    Focused,
    /// Linear array
    LinearArray,
    /// Matrix array
    MatrixArray,
    /// Planar piston
    Piston,
    /// Custom distribution
    Custom,
}

/// Pulse waveform parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulseParameters {
    /// Pulse type
    pub pulse_type: PulseType,
    /// Number of cycles
    pub cycles: f64,
    /// Envelope type
    pub envelope: EnvelopeType,
}

/// Pulse waveform types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PulseType {
    /// Continuous wave
    ContinuousWave,
    /// Tone burst
    ToneBurst,
    /// Chirp
    Chirp,
    /// Impulse
    Impulse,
    /// Custom waveform
    Custom,
}

/// Envelope types for pulses
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EnvelopeType {
    /// Rectangular window
    Rectangular,
    /// Gaussian envelope
    Gaussian,
    /// Hanning window
    Hanning,
    /// Tukey window
    Tukey,
}

impl SourceParameters {
    /// Validate source parameters
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        if self.amplitude < 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "amplitude".to_string(),
                value: self.amplitude.to_string(),
                constraint: "Must be non-negative".to_string(),
            }
            .into());
        }

        if self.frequency <= 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.radius < 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "radius".to_string(),
                value: self.radius.to_string(),
                constraint: "Must be non-negative".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for SourceParameters {
    fn default() -> Self {
        Self {
            source_type: SourceType::Point,
            amplitude: 1e6, // 1 MPa
            frequency: 1e6, // 1 MHz
            position: [0.0, 0.0, 0.0],
            radius: 1e-3, // 1mm
            delay: 0.0,
            pulse: PulseParameters::default(),
        }
    }
}

impl Default for PulseParameters {
    fn default() -> Self {
        Self {
            pulse_type: PulseType::ToneBurst,
            cycles: 3.0,
            envelope: EnvelopeType::Gaussian,
        }
    }
}
