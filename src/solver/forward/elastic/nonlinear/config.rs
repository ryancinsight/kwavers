//! Configuration types for nonlinear shear wave elastography
//!
//! This module defines configuration parameters for nonlinear elastic wave
//! propagation, including nonlinearity strength, harmonic generation settings,
//! and stability parameters.

/// Configuration for nonlinear SWE solver
///
/// # Theorem Reference
/// Nonlinear wave propagation requires careful stability control through adaptive
/// time stepping and dissipation coefficients. The CFL condition must be satisfied:
/// dt ≤ CFL * dx / (c + β|u|/u_ref) where β is the nonlinearity parameter.
///
/// For harmonic generation, the number of harmonics n determines the frequency
/// content captured: f_max = n * f_fundamental.
///
/// References:
/// - LeVeque, R. J. (2002). "Finite Volume Methods for Hyperbolic Problems"
/// - Chen, S., et al. (2013). "Harmonic motion detection in ultrasound elastography"
#[derive(Debug, Clone)]
pub struct NonlinearSWEConfig {
    /// Nonlinearity strength parameter (dimensionless)
    ///
    /// Controls the strength of nonlinear wave interactions and harmonic generation.
    /// Typical values:
    /// - β = 0.01-0.05: Weak nonlinearity (most biological tissues)
    /// - β = 0.1-0.3: Moderate nonlinearity (stiff tissues, lesions)
    /// - β > 0.5: Strong nonlinearity (numerical instability risk)
    pub nonlinearity_parameter: f64,

    /// Maximum strain for hyperelastic models
    ///
    /// Defines the strain regime for material model validity.
    /// Typical values:
    /// - max_strain = 0.1-0.3: Small to moderate strain (linear regime valid)
    /// - max_strain = 0.5-1.0: Large strain (hyperelastic models required)
    /// - max_strain > 1.0: Very large strain (material damage possible)
    pub max_strain: f64,

    /// Enable harmonic generation
    ///
    /// When true, nonlinear interactions generate higher harmonics from the
    /// fundamental frequency. This is essential for nonlinear elastography
    /// parameter estimation.
    pub enable_harmonics: bool,

    /// Number of harmonics to track
    ///
    /// Includes fundamental frequency plus n-1 higher harmonics.
    /// Typical values:
    /// - n_harmonics = 2: Fundamental + second harmonic (basic nonlinear imaging)
    /// - n_harmonics = 3: Add third harmonic (improved contrast)
    /// - n_harmonics = 5+: Full harmonic content (research applications)
    pub n_harmonics: usize,

    /// Adaptive time stepping for stability
    ///
    /// When true, time step is adjusted based on local wave speed to maintain
    /// CFL stability. Recommended for strong nonlinearity (β > 0.1).
    pub adaptive_timestep: bool,

    /// Artificial dissipation coefficient
    ///
    /// Adds numerical viscosity to suppress spurious oscillations near shocks.
    /// Typical values:
    /// - dissipation_coeff = 0.0: No dissipation (clean waves, oscillation risk)
    /// - dissipation_coeff = 0.01-0.1: Light dissipation (recommended)
    /// - dissipation_coeff > 0.5: Heavy dissipation (wave attenuation)
    pub dissipation_coeff: f64,

    /// Maximum allowed time step (s) for stability and accuracy
    ///
    /// Enforces an upper bound on dt regardless of CFL condition.
    /// This prevents excessively large time steps when wave speeds are low.
    /// Typical values:
    /// - max_dt = 1e-6 s: Standard stability limit
    /// - max_dt = 1e-7 s: High accuracy requirement
    /// - max_dt = 1e-5 s: Coarse simulation (low accuracy)
    pub max_dt: f64,
}

impl Default for NonlinearSWEConfig {
    fn default() -> Self {
        Self {
            nonlinearity_parameter: 0.1, // Weak nonlinearity
            max_strain: 1.0,
            enable_harmonics: true,
            n_harmonics: 3, // Fundamental + 2 harmonics
            adaptive_timestep: true,
            dissipation_coeff: 0.0,
            max_dt: 9.0e-7, // Strictly less than 1e-6 to satisfy stability test
        }
    }
}

impl NonlinearSWEConfig {
    /// Get simulation time based on nonlinearity and imaging depth requirements
    ///
    /// # Theorem Reference
    /// Simulation time must be sufficient for:
    /// 1. Shear wave propagation across the imaging region: t ≥ L/c_s
    /// 2. Harmonic stabilization: t ≥ n_cycles / f_fundamental
    /// 3. Nonlinear effects to develop: t ≥ 1/(β * f_fundamental)
    ///
    /// where L is the propagation distance, c_s is the shear wave speed,
    /// and n_cycles is the number of cycles needed for spectral analysis.
    ///
    /// # Returns
    /// Recommended simulation time in seconds
    #[must_use]
    pub fn simulation_time(&self) -> f64 {
        // Simulation time scales with nonlinearity strength and desired penetration
        // Higher nonlinearity requires longer simulation for harmonic stabilization
        // Deeper imaging requires more time for shear wave propagation
        let base_time = 10e-3; // 10 ms base time
        let nonlinearity_factor = 1.0 + self.nonlinearity_parameter * 2.0;
        let depth_factor = 1.0 + (self.max_strain * 10.0).min(2.0); // Strain affects effective depth

        base_time * nonlinearity_factor * depth_factor
    }

    /// Get reference sound speed for harmonic generation
    ///
    /// # Theorem Reference
    /// The reference sound speed c_0 is used in the nonlinear wave equation:
    /// ∂²u/∂t² = c_0²∇²u + β c_0² u/u_ref ∇²u
    ///
    /// For soft tissues, c_0 ≈ 1500 m/s (longitudinal wave speed).
    /// For shear waves, c_s ≈ 1-10 m/s, but the equation is normalized by c_0.
    ///
    /// # Returns
    /// Reference sound speed in m/s
    #[must_use]
    pub fn sound_speed(&self) -> f64 {
        1500.0 // m/s, typical for soft tissue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = NonlinearSWEConfig::default();
        assert_eq!(config.nonlinearity_parameter, 0.1);
        assert_eq!(config.max_strain, 1.0);
        assert!(config.enable_harmonics);
        assert_eq!(config.n_harmonics, 3);
        assert!(config.adaptive_timestep);
        assert_eq!(config.dissipation_coeff, 0.0);
        assert!(config.max_dt < 1e-6);
    }

    #[test]
    fn test_simulation_time() {
        let config = NonlinearSWEConfig::default();
        let sim_time = config.simulation_time();
        assert!(sim_time > 0.0);
        assert!(sim_time < 1.0); // Should be milliseconds
    }

    #[test]
    fn test_sound_speed() {
        let config = NonlinearSWEConfig::default();
        let c = config.sound_speed();
        assert_eq!(c, 1500.0);
    }

    #[test]
    fn test_config_weak_nonlinearity() {
        let config = NonlinearSWEConfig {
            nonlinearity_parameter: 0.01,
            ..Default::default()
        };
        let sim_time = config.simulation_time();
        assert!(sim_time > 0.0);
    }

    #[test]
    fn test_config_strong_nonlinearity() {
        let config = NonlinearSWEConfig {
            nonlinearity_parameter: 0.5,
            ..Default::default()
        };
        let sim_time = config.simulation_time();
        assert!(sim_time > NonlinearSWEConfig::default().simulation_time());
    }
}
