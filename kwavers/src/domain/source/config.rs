//! Source configuration parameters
//!
//! Defines the configuration structures for acoustic sources.

use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use crate::core::error::{ConfigError, KwaversResult};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

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
    /// Cover the full `0 <= theta <= pi/2` hemispherical focused-bowl aperture.
    Hemisphere,
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
    /// Use `position` as an axis reference rather than the bowl vertex and
    /// construct the source vertex from the explicit curvature radius.
    AxisReferencePolarBounds {
        /// Curvature radius from the acoustic focus to the bowl surface [m].
        radius_of_curvature_m: f64,
        /// Minimum polar angle from the vertex-to-focus axis [rad].
        theta_min_rad: f64,
        /// Maximum polar angle from the vertex-to-focus axis [rad].
        theta_max_rad: f64,
    },
    /// Use `position` as an axis reference and construct a hemispherical
    /// focused-bowl aperture from an explicit curvature radius.
    AxisReferenceHemisphere {
        /// Curvature radius from the acoustic focus to the bowl surface [m].
        radius_of_curvature_m: f64,
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
    /// # Theorem
    ///
    /// A source configuration accepted by this method contains only finite
    /// real-valued physical scalars, positive timing/frequency counts, and
    /// aperture bounds inside the spherical-bowl domain. Therefore downstream
    /// source constructors cannot receive `NaN`/infinite coordinates or an
    /// impossible angular aperture through this public configuration boundary.
    pub fn validate(&self) -> KwaversResult<()> {
        validate_nonnegative_finite("amplitude", self.amplitude)?;
        validate_positive_finite("frequency", self.frequency)?;
        validate_nonnegative_finite("radius", self.radius)?;
        validate_finite("phase", self.phase)?;
        validate_finite("delay", self.delay)?;
        validate_finite_vector("position", self.position)?;
        if let Some(focus) = self.focus {
            validate_finite_vector("focus", focus)?;
        }
        if let Some(num_elements) = self.num_elements {
            if num_elements == 0 {
                return Err(invalid_value(
                    "num_elements",
                    num_elements,
                    "Must be positive when provided",
                ));
            }
        }
        validate_pulse_parameters(&self.pulse)?;
        validate_focused_bowl_aperture(self.focused_bowl_aperture)?;

        Ok(())
    }
}

fn validate_pulse_parameters(pulse: &PulseParameters) -> KwaversResult<()> {
    validate_positive_finite("pulse.cycles", pulse.cycles)
}

fn validate_focused_bowl_aperture(aperture: FocusedBowlAperture) -> KwaversResult<()> {
    match aperture {
        FocusedBowlAperture::Diameter | FocusedBowlAperture::Hemisphere => Ok(()),
        FocusedBowlAperture::PolarSpan { theta_max_rad } => {
            validate_polar_bounds(0.0, theta_max_rad)
        }
        FocusedBowlAperture::PolarBounds {
            theta_min_rad,
            theta_max_rad,
        } => validate_polar_bounds(theta_min_rad, theta_max_rad),
        FocusedBowlAperture::AxisProjectionBounds {
            axis_projection_min,
            axis_projection_max,
        } => validate_axis_projection_bounds(axis_projection_min, axis_projection_max),
        FocusedBowlAperture::AxisReferencePolarBounds {
            radius_of_curvature_m,
            theta_min_rad,
            theta_max_rad,
        } => {
            validate_positive_finite(
                "focused_bowl_aperture.radius_of_curvature_m",
                radius_of_curvature_m,
            )?;
            validate_polar_bounds(theta_min_rad, theta_max_rad)
        }
        FocusedBowlAperture::AxisReferenceHemisphere {
            radius_of_curvature_m,
        } => validate_positive_finite(
            "focused_bowl_aperture.radius_of_curvature_m",
            radius_of_curvature_m,
        ),
    }
}

fn validate_polar_bounds(theta_min_rad: f64, theta_max_rad: f64) -> KwaversResult<()> {
    if theta_min_rad.is_finite()
        && theta_max_rad.is_finite()
        && theta_min_rad >= 0.0
        && theta_min_rad < theta_max_rad
        && theta_max_rad <= PI
    {
        Ok(())
    } else {
        Err(invalid_value(
            "focused_bowl_aperture.polar_bounds",
            format!("[{theta_min_rad}, {theta_max_rad}]"),
            "Must satisfy 0 <= theta_min < theta_max <= pi",
        ))
    }
}

fn validate_axis_projection_bounds(
    axis_projection_min: f64,
    axis_projection_max: f64,
) -> KwaversResult<()> {
    if axis_projection_min.is_finite()
        && axis_projection_max.is_finite()
        && axis_projection_min >= -1.0
        && axis_projection_min < axis_projection_max
        && axis_projection_max <= 1.0
    {
        Ok(())
    } else {
        Err(invalid_value(
            "focused_bowl_aperture.axis_projection_bounds",
            format!("[{axis_projection_min}, {axis_projection_max}]"),
            "Must satisfy -1 <= min < max <= 1",
        ))
    }
}

fn validate_finite_vector(parameter: &str, value: [f64; 3]) -> KwaversResult<()> {
    for (axis, component) in value.into_iter().enumerate() {
        if !component.is_finite() {
            return Err(invalid_value(
                format!("{parameter}[{axis}]"),
                component,
                "Must be finite",
            ));
        }
    }
    Ok(())
}

fn validate_positive_finite(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_value(
            parameter,
            value,
            "Must be finite and positive",
        ))
    }
}

fn validate_nonnegative_finite(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_value(
            parameter,
            value,
            "Must be finite and non-negative",
        ))
    }
}

fn validate_finite(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_value(parameter, value, "Must be finite"))
    }
}

fn invalid_value(
    parameter: impl Into<String>,
    value: impl ToString,
    constraint: impl Into<String>,
) -> crate::core::error::KwaversError {
    ConfigError::InvalidValue {
        parameter: parameter.into(),
        value: value.to_string(),
        constraint: constraint.into(),
    }
    .into()
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
