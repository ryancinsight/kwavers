//! PINN Source Adapter Layer.
//!
//! Bridges `domain::source` types to PINN-specific representations,
//! eliminating duplication while maintaining SSOT principles.
//!
//! ## Architecture
//!
//! ```text
//! Domain Layer (SSOT)         Adapter Layer              PINN Layer
//! ┌─────────────────┐        ┌──────────────┐        ┌──────────────┐
//! │ domain::source  │───────►│ PinnSource   │───────►│ PhysicsDomain│
//! │ domain::signal  │        │ Adapter      │        │ Boundary Spec│
//! └─────────────────┘        └──────────────┘        └──────────────┘
//! ```

use kwavers_core::constants::numerical::TWO_PI;
#[cfg(test)]
mod tests;

use kwavers_source::{types::SourceFocalProperties as DomainFocalProperties, Source, SourceField};
use std::sync::Arc;

/// Acoustic source specification for PINN training.
///
/// Lightweight adapter over `domain::source::Source` that extracts
/// the information needed for PINN boundary conditions and source terms.
#[derive(Debug, Clone)]
pub struct PinnAcousticSource {
    /// Reference position for the source (extracted from domain source).
    pub position: (f64, f64, f64),
    /// Source classification for PINN physics.
    pub source_class: PinnSourceClass,
    /// Frequency (Hz) — extracted from signal.
    pub frequency: f64,
    /// Peak amplitude — extracted from signal/source.
    pub amplitude: f64,
    /// Phase offset (radians).
    pub phase: f64,
    /// Optional focal properties.
    pub focal_properties: Option<PinnSourceFocalProperties>,
}

/// PINN-specific source classification.
///
/// Maps domain source types to physics-informed boundary conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnSourceClass {
    /// Point pressure source (monopole).
    Monopole,
    /// Velocity source (dipole).
    Dipole,
    /// Distributed source with focal point.
    Focused,
    /// Distributed source without focal point.
    Distributed,
}

/// Focal properties for focused sources (PINN adapter type).
///
/// Simplified version for PINN boundary conditions.
/// Complete focal properties are in `domain::source::SourceFocalProperties`.
#[derive(Debug, Clone, Copy)]
pub struct PinnSourceFocalProperties {
    /// Focal length (m).
    pub focal_length: f64,
    /// Spot size at focus (m) — beam waist or FWHM.
    pub spot_size: f64,
    /// F-number (dimensionless).
    pub f_number: Option<f64>,
    /// Focal gain (dimensionless).
    pub focal_gain: Option<f64>,
}

impl From<DomainFocalProperties> for PinnSourceFocalProperties {
    fn from(props: DomainFocalProperties) -> Self {
        Self {
            focal_length: props.focal_depth,
            spot_size: props.spot_size,
            f_number: props.f_number,
            focal_gain: props.focal_gain,
        }
    }
}

impl PinnAcousticSource {
    /// Create PINN source adapter from domain source.
    ///
    /// Extracts relevant information for PINN physics specifications.
    /// # Errors
    /// - Returns [`AdapterError::NoSourcePositions`] if the source has no positions.
    ///
    pub fn from_domain_source(source: &dyn Source, time_sample: f64) -> Result<Self, AdapterError> {
        let positions = source.positions();
        if positions.is_empty() {
            return Err(AdapterError::NoSourcePositions);
        }
        let position = positions[0];

        let source_class = Self::classify_source(source, &positions);

        let signal = source.signal();
        let frequency = signal.frequency(time_sample);
        let phase = signal.phase(time_sample);

        let t_peak = (std::f64::consts::PI / 2.0 - phase) / (TWO_PI * frequency);
        let amplitude = signal.amplitude(t_peak).abs();

        let focal_properties = Self::extract_focal_properties(source);

        Ok(Self {
            position,
            source_class,
            frequency,
            amplitude,
            phase,
            focal_properties,
        })
    }

    /// Classify domain source for PINN physics.
    fn classify_source(source: &dyn Source, positions: &[(f64, f64, f64)]) -> PinnSourceClass {
        match source.source_type() {
            SourceField::Pressure => {
                if positions.len() == 1 {
                    PinnSourceClass::Monopole
                } else {
                    PinnSourceClass::Distributed
                }
            }
            SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ => {
                PinnSourceClass::Dipole
            }
        }
    }

    /// Extract focal properties from domain source via `Source::get_focal_properties`.
    fn extract_focal_properties(source: &dyn Source) -> Option<PinnSourceFocalProperties> {
        source.get_focal_properties().map(|props| props.into())
    }

    /// Source term coefficient S(t) = amplitude · cos(ωt + φ).
    pub fn source_term_coefficient(&self, t: f64) -> f64 {
        let omega = TWO_PI * self.frequency;
        self.amplitude * (omega * t + self.phase).cos()
    }

    /// Check if point (x, y, z) is within `tolerance` of the source position.
    pub fn is_near_position(&self, x: f64, y: f64, z: f64, tolerance: f64) -> bool {
        let dx = x - self.position.0;
        let dy = y - self.position.1;
        let dz = z - self.position.2;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        dist_sq <= tolerance * tolerance
    }
}

/// Adapter errors.
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("Source has no positions defined")]
    NoSourcePositions,

    #[error("Incompatible source type for PINN physics: {0}")]
    IncompatibleSourceType(String),

    #[error("Missing required source metadata: {0}")]
    MissingMetadata(String),
}

/// Convert multiple domain sources to PINN source specifications.
/// # Errors
/// - Propagates [`AdapterError`] from [`PinnAcousticSource::from_domain_source`].
///
pub fn adapt_sources(
    sources: &[Arc<dyn Source>],
    time_sample: f64,
) -> Result<Vec<PinnAcousticSource>, AdapterError> {
    sources
        .iter()
        .map(|s| PinnAcousticSource::from_domain_source(s.as_ref(), time_sample))
        .collect()
}
