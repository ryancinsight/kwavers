//! PINN Electromagnetic Source Adapter
//!
//! Adapts `domain::source::electromagnetic` types to PINN-specific representations,
//! eliminating duplication while maintaining SSOT principles.

use crate::domain::source::electromagnetic::PointEMSource;
use std::fmt::Debug;

/// Electromagnetic source specification for PINN training
///
/// Lightweight adapter over `domain::source::electromagnetic::EMSource` that extracts
/// information needed for PINN Maxwell's equations boundary conditions.
#[derive(Debug, Clone)]
pub struct PinnEMSource {
    /// Source position (x, y, z)
    pub position: (f64, f64, f64),
    /// Current density vector [Jx, Jy, Jz] (A/m²)
    pub current_density: [f64; 3],
    /// Source spatial extent (m)
    pub spatial_extent: f64,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Peak amplitude (A/m² or V/m depending on source type)
    pub amplitude: f64,
    /// Phase offset (radians)
    pub phase: f64,
}

impl PinnEMSource {
    /// Create PINN EM source adapter from domain EM source
    ///
    /// # Parameters
    /// - `source`: Domain electromagnetic source (SSOT)
    /// - `time_sample`: Time at which to sample source properties
    pub fn from_domain_source(
        source: &PointEMSource,
        _time_sample: f64,
    ) -> Result<Self, EMAdapterError> {
        // Extract position
        let position = (source.position[0], source.position[1], source.position[2]);

        // Extract frequency and temporal properties
        let frequency = source.frequency;
        let amplitude = source.amplitude;
        let phase = source.phase;

        // Compute current density from polarization and amplitude
        // For a point source, current density aligns with polarization direction
        let current_density = Self::compute_current_density(source, amplitude);

        // Spatial extent for point source (effectively a delta function)
        let spatial_extent = 0.0;

        Ok(Self {
            position,
            current_density,
            spatial_extent,
            frequency,
            amplitude,
            phase,
        })
    }

    /// Compute current density vector from source polarization
    fn compute_current_density(source: &PointEMSource, amplitude: f64) -> [f64; 3] {
        use crate::domain::source::Polarization;

        match source.polarization {
            Polarization::LinearX => [amplitude, 0.0, 0.0],
            Polarization::LinearY => [0.0, amplitude, 0.0],
            Polarization::LinearZ => [0.0, 0.0, amplitude],
            Polarization::RightCircular => {
                // Right circular: Jx = A, Jy = -iA (90° phase shift)
                // For real representation at t=0: [A, 0, 0] with phase understanding
                [amplitude, 0.0, 0.0]
            }
            Polarization::LeftCircular => {
                // Left circular: Jx = A, Jy = iA (-90° phase shift)
                [amplitude, 0.0, 0.0]
            }
            Polarization::Elliptical { ratio, phase_diff } => {
                // Elliptical polarization with specified ratio and phase
                let jy = amplitude * ratio * phase_diff.cos();
                [amplitude, jy, 0.0]
            }
        }
    }

    /// Get source term coefficient for PINN PDE residual
    ///
    /// Returns time-varying source term: J(t) = J₀ * exp(i(2πft + φ))
    pub fn source_term_coefficient(&self, t: f64) -> [f64; 3] {
        let omega = 2.0 * std::f64::consts::PI * self.frequency;
        let time_factor = (omega * t + self.phase).cos();

        [
            self.current_density[0] * time_factor,
            self.current_density[1] * time_factor,
            self.current_density[2] * time_factor,
        ]
    }

    /// Check if point is near source (for boundary condition application)
    pub fn is_near_position(&self, x: f64, y: f64, z: f64, tolerance: f64) -> bool {
        let dx = x - self.position.0;
        let dy = y - self.position.1;
        let dz = z - self.position.2;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        dist_sq <= tolerance * tolerance
    }

    /// Get current density magnitude
    pub fn current_density_magnitude(&self) -> f64 {
        let [jx, jy, jz] = self.current_density;
        (jx * jx + jy * jy + jz * jz).sqrt()
    }
}

/// EM adapter errors
#[derive(Debug, thiserror::Error)]
pub enum EMAdapterError {
    #[error("Invalid source configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Missing required EM source metadata: {0}")]
    MissingMetadata(String),

    #[error("Incompatible EM source type for PINN: {0}")]
    IncompatibleSourceType(String),
}

/// Convert multiple domain EM sources to PINN EM source specifications
pub fn adapt_em_sources(
    sources: &[PointEMSource],
    time_sample: f64,
) -> Result<Vec<PinnEMSource>, EMAdapterError> {
    sources
        .iter()
        .map(|s| PinnEMSource::from_domain_source(s, time_sample))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::source::Polarization;

    #[test]
    fn test_point_em_source_adapter() {
        // Create domain EM point source
        let domain_source = PointEMSource {
            position: [0.01, 0.02, 0.03],
            polarization: Polarization::LinearX,
            frequency: 1e9, // 1 GHz
            amplitude: 100.0,
            phase: 0.0,
        };

        // Adapt to PINN format
        let pinn_source = PinnEMSource::from_domain_source(&domain_source, 0.0)
            .expect("Should adapt successfully");

        // Verify position preserved
        assert_eq!(pinn_source.position, (0.01, 0.02, 0.03));

        // Verify current density for x-polarization
        assert!((pinn_source.current_density[0] - 100.0).abs() < 1e-6);
        assert!(pinn_source.current_density[1].abs() < 1e-6);
        assert!(pinn_source.current_density[2].abs() < 1e-6);

        // Verify frequency
        assert!((pinn_source.frequency - 1e9).abs() < 1e-6);
    }

    #[test]
    fn test_source_term_coefficient() {
        let pinn_source = PinnEMSource {
            position: (0.0, 0.0, 0.0),
            current_density: [100.0, 0.0, 0.0],
            spatial_extent: 0.0,
            frequency: 1e9,
            amplitude: 100.0,
            phase: 0.0,
        };

        // At t=0, should be J₀ * cos(0) = J₀
        let coeff_t0 = pinn_source.source_term_coefficient(0.0);
        assert!((coeff_t0[0] - 100.0).abs() < 1e-6);
        assert!(coeff_t0[1].abs() < 1e-6);
        assert!(coeff_t0[2].abs() < 1e-6);

        // At quarter period, should be near zero
        let t_quarter = 0.25 / 1e9;
        let coeff_quarter = pinn_source.source_term_coefficient(t_quarter);
        assert!(coeff_quarter[0].abs() < 1e-6);
    }

    #[test]
    fn test_y_polarization() {
        let domain_source = PointEMSource {
            position: [0.0, 0.0, 0.0],
            polarization: Polarization::LinearY,
            frequency: 1e9,
            amplitude: 50.0,
            phase: 0.0,
        };

        let pinn_source = PinnEMSource::from_domain_source(&domain_source, 0.0)
            .expect("Should adapt successfully");

        // Y-polarized: current density in y-direction
        assert!(pinn_source.current_density[0].abs() < 1e-6);
        assert!((pinn_source.current_density[1] - 50.0).abs() < 1e-6);
        assert!(pinn_source.current_density[2].abs() < 1e-6);
    }

    #[test]
    fn test_current_density_magnitude() {
        let pinn_source = PinnEMSource {
            position: (0.0, 0.0, 0.0),
            current_density: [3.0, 4.0, 0.0],
            spatial_extent: 0.0,
            frequency: 1e9,
            amplitude: 5.0,
            phase: 0.0,
        };

        let magnitude = pinn_source.current_density_magnitude();
        assert!((magnitude - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_near_position() {
        let pinn_source = PinnEMSource {
            position: (0.0, 0.0, 0.0),
            current_density: [100.0, 0.0, 0.0],
            spatial_extent: 0.0,
            frequency: 1e9,
            amplitude: 100.0,
            phase: 0.0,
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
        let source1 = PointEMSource {
            position: [0.0, 0.0, 0.0],
            polarization: Polarization::LinearX,
            frequency: 1e9,
            amplitude: 100.0,
            phase: 0.0,
        };

        let source2 = PointEMSource {
            position: [0.01, 0.0, 0.0],
            polarization: Polarization::LinearY,
            frequency: 2e9,
            amplitude: 50.0,
            phase: 0.0,
        };

        let sources = vec![source1, source2];
        let pinn_sources = adapt_em_sources(&sources, 0.0).expect("Should adapt all sources");

        assert_eq!(pinn_sources.len(), 2);
        assert!((pinn_sources[0].frequency - 1e9).abs() < 1e-6);
        assert!((pinn_sources[1].frequency - 2e9).abs() < 1e-6);
    }
}
