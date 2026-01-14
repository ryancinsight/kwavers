//! PINN Source Adapter Layer
//!
//! This module provides adapters that bridge `domain::source` types to PINN-specific
//! representations, eliminating duplication while maintaining SSOT principles.
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
//!
//! ## Design Principles
//!
//! 1. **SSOT Enforcement**: Domain types are the canonical source of truth
//! 2. **Unidirectional Dependency**: PINN layer depends on domain, never reverse
//! 3. **Thin Adaptation**: Minimal logic, primarily type conversion
//! 4. **Zero Duplication**: No domain concepts redefined in PINN layer

use crate::domain::source::{FocalProperties as DomainFocalProperties, Source, SourceField};
use std::sync::Arc;

/// Acoustic source specification for PINN training
///
/// This is a lightweight adapter over `domain::source::Source` that extracts
/// the information needed for PINN boundary conditions and source terms.
#[derive(Debug, Clone)]
pub struct PinnAcousticSource {
    /// Reference position for the source (extracted from domain source)
    pub position: (f64, f64, f64),
    /// Source classification for PINN physics
    pub source_class: PinnSourceClass,
    /// Frequency (Hz) - extracted from signal
    pub frequency: f64,
    /// Peak amplitude - extracted from signal/source
    pub amplitude: f64,
    /// Phase offset (radians)
    pub phase: f64,
    /// Optional focal properties
    pub focal_properties: Option<FocalProperties>,
}

/// PINN-specific source classification
///
/// Maps domain source types to physics-informed boundary conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnSourceClass {
    /// Point pressure source (monopole)
    Monopole,
    /// Velocity source (dipole)
    Dipole,
    /// Distributed source with focal point
    Focused,
    /// Distributed source without focal point
    Distributed,
}

/// Focal properties for focused sources (PINN adapter type)
///
/// This is a simplified version for PINN boundary conditions.
/// The complete focal properties are available in `domain::source::FocalProperties`.
#[derive(Debug, Clone, Copy)]
pub struct FocalProperties {
    /// Focal length (m)
    pub focal_length: f64,
    /// Spot size at focus (m) - beam waist or FWHM
    pub spot_size: f64,
    /// F-number (dimensionless)
    pub f_number: Option<f64>,
    /// Focal gain (dimensionless)
    pub focal_gain: Option<f64>,
}

impl From<DomainFocalProperties> for FocalProperties {
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
    /// Create PINN source adapter from domain source
    ///
    /// Extracts relevant information for PINN physics specifications.
    ///
    /// # Parameters
    /// - `source`: Domain source (SSOT)
    /// - `time_sample`: Time at which to sample signal properties
    ///
    /// # Example
    /// ```ignore
    /// use kwavers::domain::source::PointSource;
    /// use kwavers::analysis::ml::pinn::adapters::PinnAcousticSource;
    ///
    /// let domain_source: Arc<dyn Source> = /* ... */;
    /// let pinn_source = PinnAcousticSource::from_domain_source(domain_source, 0.0)?;
    /// ```
    pub fn from_domain_source(source: &dyn Source, time_sample: f64) -> Result<Self, AdapterError> {
        // Extract position (use first position for point-like sources)
        let positions = source.positions();
        if positions.is_empty() {
            return Err(AdapterError::NoSourcePositions);
        }
        let position = positions[0];

        // Classify source based on type and geometry
        let source_class = Self::classify_source(source, &positions);

        // Extract signal properties
        let signal = source.signal();
        let frequency = signal.frequency(time_sample);
        let phase = signal.phase(time_sample);

        // Extract amplitude by sampling at quarter period where sin(ωt + φ) = 1
        // This gives us the peak amplitude of the carrier signal
        let _period = if frequency > 0.0 {
            1.0 / frequency
        } else {
            1.0
        };
        let t_peak =
            (std::f64::consts::PI / 2.0 - phase) / (2.0 * std::f64::consts::PI * frequency);
        let amplitude = signal.amplitude(t_peak).abs();

        // Extract focal properties if applicable
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

    /// Classify domain source for PINN physics
    fn classify_source(source: &dyn Source, positions: &[(f64, f64, f64)]) -> PinnSourceClass {
        match source.source_type() {
            SourceField::Pressure => {
                // Pressure sources map to monopole or distributed
                if positions.len() == 1 {
                    PinnSourceClass::Monopole
                } else {
                    // Check if focused (would need additional metadata)
                    // For now, default to distributed
                    PinnSourceClass::Distributed
                }
            }
            SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ => {
                // Velocity sources map to dipole
                PinnSourceClass::Dipole
            }
        }
    }

    /// Extract focal properties from domain source
    ///
    /// Uses the `Source` trait's focal property methods to extract focusing characteristics.
    /// Returns `None` for unfocused sources (plane waves, point sources, etc.)
    fn extract_focal_properties(source: &dyn Source) -> Option<FocalProperties> {
        // Use the Source trait's get_focal_properties() method
        source.get_focal_properties().map(|props| props.into())
    }

    /// Convert to source term coefficient for PINN PDE residual
    ///
    /// Returns the source term multiplier: `S(t) = amplitude * exp(i(2πft + φ))`
    pub fn source_term_coefficient(&self, t: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * self.frequency;
        self.amplitude * (omega * t + self.phase).cos()
    }

    /// Check if point is near source (for boundary condition application)
    pub fn is_near_position(&self, x: f64, y: f64, z: f64, tolerance: f64) -> bool {
        let dx = x - self.position.0;
        let dy = y - self.position.1;
        let dz = z - self.position.2;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        dist_sq <= tolerance * tolerance
    }
}

/// Adapter errors
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("Source has no positions defined")]
    NoSourcePositions,

    #[error("Incompatible source type for PINN physics: {0}")]
    IncompatibleSourceType(String),

    #[error("Missing required source metadata: {0}")]
    MissingMetadata(String),
}

/// Convert multiple domain sources to PINN source specifications
pub fn adapt_sources(
    sources: &[Arc<dyn Source>],
    time_sample: f64,
) -> Result<Vec<PinnAcousticSource>, AdapterError> {
    sources
        .iter()
        .map(|s| PinnAcousticSource::from_domain_source(s.as_ref(), time_sample))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::signal::waveform::SineWave;
    use crate::domain::source::PointSource;

    #[test]
    fn test_point_source_adapter() {
        // Create domain point source (SSOT)
        let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let position = (0.01, 0.02, 0.03);
        let domain_source = PointSource::new(position, signal);

        // Adapt to PINN format
        let pinn_source = PinnAcousticSource::from_domain_source(&domain_source, 0.0)
            .expect("Should adapt successfully");

        // Verify position preserved
        assert_eq!(pinn_source.position, position);

        // Verify classification
        assert_eq!(pinn_source.source_class, PinnSourceClass::Monopole);

        // Verify signal properties
        assert!((pinn_source.frequency - 1e6).abs() < 1e-6);
        assert!((pinn_source.amplitude - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_source_term_coefficient() {
        let pinn_source = PinnAcousticSource {
            position: (0.0, 0.0, 0.0),
            source_class: PinnSourceClass::Monopole,
            frequency: 1e6,
            amplitude: 100.0,
            phase: 0.0,
            focal_properties: None,
        };

        // At t=0, should be amplitude * cos(0) = amplitude
        let coeff_t0 = pinn_source.source_term_coefficient(0.0);
        assert!((coeff_t0 - 100.0).abs() < 1e-6);

        // At quarter period, should be near zero
        let t_quarter = 0.25 / 1e6;
        let coeff_quarter = pinn_source.source_term_coefficient(t_quarter);
        assert!(coeff_quarter.abs() < 1e-6);
    }

    #[test]
    fn test_is_near_position() {
        let pinn_source = PinnAcousticSource {
            position: (0.0, 0.0, 0.0),
            source_class: PinnSourceClass::Monopole,
            frequency: 1e6,
            amplitude: 1.0,
            phase: 0.0,
            focal_properties: None,
        };

        // Point at source
        assert!(pinn_source.is_near_position(0.0, 0.0, 0.0, 1e-3));

        // Point within tolerance
        assert!(pinn_source.is_near_position(0.0005, 0.0, 0.0, 1e-3));

        // Point outside tolerance
        assert!(!pinn_source.is_near_position(0.002, 0.0, 0.0, 1e-3));
    }

    #[test]
    fn test_adapt_multiple_sources() {
        let signal1 = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let signal2 = Arc::new(SineWave::new(2e6, 2.0, 0.0));

        let source1: Arc<dyn Source> = Arc::new(PointSource::new((0.0, 0.0, 0.0), signal1));
        let source2: Arc<dyn Source> = Arc::new(PointSource::new((0.01, 0.0, 0.0), signal2));

        let sources = vec![source1, source2];
        let pinn_sources = adapt_sources(&sources, 0.0).expect("Should adapt all sources");

        assert_eq!(pinn_sources.len(), 2);
        assert!((pinn_sources[0].frequency - 1e6).abs() < 1e-6);
        assert!((pinn_sources[1].frequency - 2e6).abs() < 1e-6);
    }

    #[test]
    fn test_focal_properties_extraction() {
        use crate::domain::source::wavefront::gaussian::{GaussianConfig, GaussianSource};

        // Create a focused Gaussian source
        let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let config = GaussianConfig {
            focal_point: (0.0, 0.0, 0.05), // 5cm focal depth
            waist_radius: 1e-3,            // 1mm waist
            wavelength: 1.5e-3,            // 1.5mm (1MHz in water)
            direction: (0.0, 0.0, 1.0),
            ..Default::default()
        };
        let gaussian_source = GaussianSource::new(config, signal);

        // Adapt to PINN format
        let pinn_source = PinnAcousticSource::from_domain_source(&gaussian_source, 0.0)
            .expect("Should adapt Gaussian source");

        // Verify focal properties were extracted
        assert!(
            pinn_source.focal_properties.is_some(),
            "Gaussian source should have focal properties"
        );

        let focal_props = pinn_source.focal_properties.unwrap();

        // Check focal length is approximately 5cm
        assert!(
            (focal_props.focal_length - 0.05).abs() < 1e-3,
            "Focal length should be ~5cm, got {}",
            focal_props.focal_length
        );

        // Check spot size is 1mm (waist radius)
        assert!(
            (focal_props.spot_size - 1e-3).abs() < 1e-6,
            "Spot size should be 1mm, got {}",
            focal_props.spot_size
        );

        // Check F-number exists
        assert!(
            focal_props.f_number.is_some(),
            "F-number should be available"
        );

        // Check focal gain exists
        assert!(
            focal_props.focal_gain.is_some(),
            "Focal gain should be available"
        );
    }

    #[test]
    fn test_unfocused_source_no_focal_properties() {
        let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let point_source = PointSource::new((0.0, 0.0, 0.0), signal);

        let pinn_source = PinnAcousticSource::from_domain_source(&point_source, 0.0)
            .expect("Should adapt point source");

        // Point sources should not have focal properties
        assert!(
            pinn_source.focal_properties.is_none(),
            "Point source should not have focal properties"
        );
    }
}
