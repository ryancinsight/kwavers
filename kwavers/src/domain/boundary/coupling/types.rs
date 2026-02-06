//! Shared types and trait definitions for boundary coupling
//!
//! This module provides the common type definitions and trait re-exports
//! used across all coupling boundary condition implementations.

// Re-export commonly used types for public API
pub use crate::domain::boundary::traits::BoundaryDirections;

/// Physics domain identifier for multi-physics coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsDomain {
    /// Acoustic wave propagation
    Acoustic,
    /// Elastic wave propagation (solid mechanics)
    Elastic,
    /// Electromagnetic wave propagation
    Electromagnetic,
    /// Heat transfer and thermal diffusion
    Thermal,
    /// Custom physics domain (user-defined)
    Custom(u32),
}

/// Type of coupling between physics domains
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingType {
    /// Acoustic-elastic coupling (fluid-structure interaction)
    AcousticElastic,
    /// Electromagnetic-acoustic coupling (optoacoustic, photoacoustic)
    ElectromagneticAcoustic {
        /// Optical absorption coefficient (m⁻¹)
        optical_absorption: f64,
    },
    /// Acoustic-thermal coupling (heat generation from absorption)
    AcousticThermal,
    /// Electromagnetic-thermal coupling (Joule heating)
    ElectromagneticThermal,
    /// Custom coupling type (user-defined)
    Custom(String),
}

/// Frequency profile for frequency-dependent boundary conditions
#[derive(Debug, Clone, PartialEq, Default)]
pub enum FrequencyProfile {
    /// Flat frequency response (constant across all frequencies)
    #[default]
    Flat,
    /// Gaussian frequency profile (peaked at center frequency)
    Gaussian {
        /// Center frequency (Hz)
        center_freq: f64,
        /// Bandwidth (Hz)
        bandwidth: f64,
    },
    /// Custom frequency profile (user-defined function)
    Custom(Vec<(f64, f64)>), // (frequency, value) pairs
}

/// Transmission condition for domain decomposition
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TransmissionCondition {
    /// Dirichlet transmission: u_interface = u_neighbor
    #[default]
    Dirichlet,
    /// Neumann transmission: ∂u₁/∂n = ∂u₂/∂n (flux continuity)
    Neumann,
    /// Robin transmission: ∂u/∂n + αu = β
    Robin {
        /// Robin coefficient α (coupling strength)
        alpha: f64,
        /// Robin coefficient β (external source term)
        beta: f64,
    },
    /// Optimized Schwarz with relaxation parameter
    Optimized,
}

impl FrequencyProfile {
    /// Evaluate frequency profile at given frequency
    ///
    /// # Arguments
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    /// Profile value at the given frequency (dimensionless scaling factor)
    pub fn evaluate(&self, frequency: f64) -> f64 {
        match self {
            Self::Flat => 1.0,
            Self::Gaussian {
                center_freq,
                bandwidth,
            } => {
                let df = frequency - center_freq;
                let sigma = bandwidth / (2.0 * (2.0_f64.ln()).sqrt());
                (-df * df / (2.0 * sigma * sigma)).exp()
            }
            Self::Custom(points) => {
                // Linear interpolation between points
                if points.is_empty() {
                    return 1.0;
                }
                if frequency <= points[0].0 {
                    return points[0].1;
                }
                if frequency >= points[points.len() - 1].0 {
                    return points[points.len() - 1].1;
                }

                // Binary search for interpolation interval
                let idx = points
                    .binary_search_by(|p| p.0.partial_cmp(&frequency).unwrap())
                    .unwrap_or_else(|i| i);

                if idx == 0 {
                    return points[0].1;
                }

                let (f0, v0) = points[idx - 1];
                let (f1, v1) = points[idx];
                let t = (frequency - f0) / (f1 - f0);
                v0 + t * (v1 - v0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_profile_flat() {
        let profile = FrequencyProfile::Flat;
        assert_eq!(profile.evaluate(0.0), 1.0);
        assert_eq!(profile.evaluate(1e6), 1.0);
    }

    #[test]
    fn test_frequency_profile_gaussian() {
        let profile = FrequencyProfile::Gaussian {
            center_freq: 1e6,
            bandwidth: 0.5e6,
        };

        // Peak at center frequency
        assert!((profile.evaluate(1e6) - 1.0).abs() < 1e-10);

        // Lower at off-center frequencies
        assert!(profile.evaluate(0.5e6) < 1.0);
        assert!(profile.evaluate(1.5e6) < 1.0);

        // Symmetric
        let v1 = profile.evaluate(0.8e6);
        let v2 = profile.evaluate(1.2e6);
        assert!((v1 - v2).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_profile_custom() {
        let profile =
            FrequencyProfile::Custom(vec![(0.0, 0.0), (1e6, 1.0), (2e6, 0.5), (3e6, 0.0)]);

        // Exact points
        assert_eq!(profile.evaluate(0.0), 0.0);
        assert_eq!(profile.evaluate(1e6), 1.0);
        assert_eq!(profile.evaluate(2e6), 0.5);
        assert_eq!(profile.evaluate(3e6), 0.0);

        // Interpolation
        assert!((profile.evaluate(0.5e6) - 0.5).abs() < 1e-10);
        assert!((profile.evaluate(1.5e6) - 0.75).abs() < 1e-10);

        // Extrapolation (clamp to edges)
        assert_eq!(profile.evaluate(-1.0), 0.0);
        assert_eq!(profile.evaluate(4e6), 0.0);
    }

    #[test]
    fn test_transmission_condition_default() {
        let cond = TransmissionCondition::default();
        assert_eq!(cond, TransmissionCondition::Dirichlet);
    }
}
