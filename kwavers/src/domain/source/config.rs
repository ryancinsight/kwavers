//! Source configuration parameters
//!
//! Defines the configuration structures for acoustic sources.

use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use serde::{Deserialize, Serialize};

use crate::domain::source::types::SourceField;

/// Source configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSourceParameters {
    /// Source model (geometry/distribution)
    pub model: SourceModel,
    /// Field type to inject (Pressure, Velocity, etc.)
    #[serde(default)]
    pub source_field: SourceField,
    /// Source amplitude in Pa
    pub amplitude: f64,
    /// Source frequency in Hz
    pub frequency: f64,
    /// Source phase in radians
    #[serde(default)]
    pub phase: f64,
    /// Source position [x, y, z] in meters (Center point)
    pub position: [f64; 3],
    /// Source size/radius in meters (for Piston, Gaussian waist, etc.)
    pub radius: f64,
    /// Focal point [x, y, z] (for focused sources) or Direction [x, y, z] (for plane waves)
    pub focus: Option<[f64; 3]>,
    /// Number of elements (for arrays)
    pub num_elements: Option<usize>,
    /// Aperture parameterization for focused bowl sources.
    #[serde(default)]
    pub focused_bowl_aperture: FocusedBowlAperture,
    /// Time delay in seconds
    pub delay: f64,
    /// Pulse parameters
    pub pulse: PulseParameters,
}

/// Source-domain aperture selector for focused bowl transducers.
///
/// `Diameter` preserves the historical meaning of [`DomainSourceParameters::radius`]
/// as the projected aperture radius. The angular variants route configured
/// simulations through the same [`crate::domain::source::transducers::focused::BowlTransducer`]
/// constructors used by direct Rust callers, avoiding duplicated clinical
/// geometry outside the source boundary.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FocusedBowlAperture {
    /// Use `DomainSourceParameters::radius` as the projected aperture radius.
    #[default]
    Diameter,
    /// Cover `0 <= theta <= theta_max_rad`.
    PolarSpan {
        /// Maximum polar angle from the vertex-to-focus axis [rad].
        theta_max_rad: f64,
    },
    /// Cover `theta_min_rad <= theta <= theta_max_rad`.
    PolarBounds {
        /// Minimum polar angle from the vertex-to-focus axis [rad].
        theta_min_rad: f64,
        /// Maximum polar angle from the vertex-to-focus axis [rad].
        theta_max_rad: f64,
    },
    /// Cover normalized aperture-axis projection bounds.
    AxisProjectionBounds {
        /// Lower normalized aperture-axis projection.
        axis_projection_min: f64,
        /// Upper normalized aperture-axis projection.
        axis_projection_max: f64,
    },
}

/// Types of acoustic sources (Geometry/Distribution)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceModel {
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
    /// Plane wave
    PlaneWave,
    /// Gaussian beam
    Gaussian,
    /// Bessel beam
    Bessel,
    /// Spherical wave
    Spherical,
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
    /// Continuous sine wave
    ContinuousWave,
    /// Sine wave (alias for ContinuousWave)
    Sine,
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
    Hann,
    /// Hanning window (alias)
    Hanning,
    /// Tukey window
    Tukey,
    /// Blackman window
    Blackman,
    /// Hamming window
    Hamming,
}

impl DomainSourceParameters {
    /// Validate source parameters
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        if self.amplitude < 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "amplitude".to_owned(),
                value: self.amplitude.to_string(),
                constraint: "Must be non-negative".to_owned(),
            }
            .into());
        }

        if self.frequency <= 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "frequency".to_owned(),
                value: self.frequency.to_string(),
                constraint: "Must be positive".to_owned(),
            }
            .into());
        }

        if self.radius < 0.0 {
            return Err(crate::core::error::ConfigError::InvalidValue {
                parameter: "radius".to_owned(),
                value: self.radius.to_string(),
                constraint: "Must be non-negative".to_owned(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for DomainSourceParameters {
    fn default() -> Self {
        Self {
            model: SourceModel::Point,
            source_field: SourceField::default(),
            amplitude: MPA_TO_PA, // 1 MPa
            frequency: MHZ_TO_HZ, // 1 MHz
            phase: 0.0,
            position: [0.0, 0.0, 0.0],
            radius: 1e-3, // 1mm
            focus: None,
            num_elements: None,
            focused_bowl_aperture: FocusedBowlAperture::default(),
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
