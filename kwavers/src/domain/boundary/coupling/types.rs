//! Shared types and trait definitions for boundary coupling
//!
//! This module provides the common type definitions and trait re-exports
//! used across all coupling boundary condition implementations.

// Re-export commonly used types for public API
pub use crate::domain::boundary::traits::BoundaryDirections;

/// Physics domain identifier for multi-physics coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCouplingPhysicsDomain {
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
pub enum BoundaryCouplingType {
    /// Acoustic-elastic coupling (fluid-structure interaction).
    ///
    /// Carries the specific acoustic impedances Z = ρc [Pa·s/m = Rayl] of the
    /// two media on either side of the interface. These are required to compute
    /// the plane-wave power transmission coefficient
    ///
    /// ```text
    /// τ = 4 Z₁ Z₂ / (Z₁ + Z₂)²    (Brekhovskikh & Godin 1998, §1.5)
    /// ```
    ///
    /// Typical values:
    /// - Water:       Z ≈ 1.479×10⁶ Rayl  (ρ = 998 kg/m³, c = 1482 m/s)
    /// - Soft tissue: Z ≈ 1.632×10⁶ Rayl  (ρ = 1060 kg/m³, c = 1540 m/s)
    /// - Cortical bone: Z ≈ 6.26×10⁶ Rayl (ρ = 1900 kg/m³, c = 3294 m/s)
    AcousticElastic {
        /// Specific acoustic impedance of medium 1 [Pa·s/m = Rayl]
        z1_rayl: f64,
        /// Specific acoustic impedance of medium 2 [Pa·s/m = Rayl]
        z2_rayl: f64,
    },
    /// Electromagnetic-acoustic coupling (optoacoustic, photoacoustic)
    ElectromagneticAcoustic {
        /// Optical absorption coefficient (m⁻¹)
        optical_absorption: f64,
        /// Grüneisen parameter Γ = β c² / c_p (dimensionless).
        /// Water: Γ ≈ 0.12 (20°C); soft tissue: Γ ≈ 0.15 (37°C).
        /// Reference: Xu & Wang (2006), Rev. Sci. Instrum. 77, 041101.
        gruneisen: f64,
    },
    /// Acoustic-thermal coupling (heat generation from absorption).
    ///
    /// Carries the acoustic absorption coefficient α [Np/m] and the
    /// medium density ρ [kg/m³] and specific heat c_p [J/(kg·K)] needed
    /// to compute the volumetric heat source Q = 2αI/(ρc_p).
    AcousticThermal {
        /// Acoustic amplitude absorption coefficient [Np/m]
        alpha_np_per_m: f64,
        /// Density [kg/m³]
        rho_kg_per_m3: f64,
        /// Specific heat at constant pressure [J/(kg·K)]
        c_p_j_per_kg_k: f64,
    },
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
pub enum BoundaryTransmissionCondition {
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
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
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
                    .binary_search_by(|p| p.0.total_cmp(&frequency))
                    .unwrap_or_else(|i| i);

                if idx == 0 {
                    return points[0].1;
                }

                let (f0, v0) = points[idx - 1];
                let (f1, v1) = points[idx];
                let t = (frequency - f0) / (f1 - f0);
                t.mul_add(v1 - v0, v0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::numerical::MHZ_TO_HZ;

    #[test]
    fn test_frequency_profile_flat() {
        let profile = FrequencyProfile::Flat;
        assert_eq!(profile.evaluate(0.0), 1.0);
        assert_eq!(profile.evaluate(MHZ_TO_HZ), 1.0);
    }

    #[test]
    fn test_frequency_profile_gaussian() {
        let profile = FrequencyProfile::Gaussian {
            center_freq: MHZ_TO_HZ,
            bandwidth: 0.5 * MHZ_TO_HZ,
        };

        // Peak at center frequency
        assert!((profile.evaluate(MHZ_TO_HZ) - 1.0).abs() < 1e-10);

        // Lower at off-center frequencies
        assert!(profile.evaluate(0.5 * MHZ_TO_HZ) < 1.0);
        assert!(profile.evaluate(1.5 * MHZ_TO_HZ) < 1.0);

        // Symmetric
        let v1 = profile.evaluate(0.8 * MHZ_TO_HZ);
        let v2 = profile.evaluate(1.2 * MHZ_TO_HZ);
        assert!((v1 - v2).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_profile_custom() {
        let profile = FrequencyProfile::Custom(vec![
            (0.0, 0.0),
            (MHZ_TO_HZ, 1.0),
            (2.0 * MHZ_TO_HZ, 0.5),
            (3.0 * MHZ_TO_HZ, 0.0),
        ]);

        // Exact points
        assert_eq!(profile.evaluate(0.0), 0.0);
        assert_eq!(profile.evaluate(MHZ_TO_HZ), 1.0);
        assert_eq!(profile.evaluate(2.0 * MHZ_TO_HZ), 0.5);
        assert_eq!(profile.evaluate(3.0 * MHZ_TO_HZ), 0.0);

        // Interpolation
        assert!((profile.evaluate(0.5 * MHZ_TO_HZ) - 0.5).abs() < 1e-10);
        assert!((profile.evaluate(1.5 * MHZ_TO_HZ) - 0.75).abs() < 1e-10);

        // Extrapolation (clamp to edges)
        assert_eq!(profile.evaluate(-1.0), 0.0);
        assert_eq!(profile.evaluate(4.0 * MHZ_TO_HZ), 0.0);
    }

    #[test]
    fn test_transmission_condition_default() {
        let cond = BoundaryTransmissionCondition::default();
        assert_eq!(cond, BoundaryTransmissionCondition::Dirichlet);
    }
}
