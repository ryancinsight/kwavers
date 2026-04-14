use crate::core::constants::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, GRUNEISEN_WATER_37C,
    OPTICAL_ABSORPTION_TISSUE_NIR, REDUCED_SCATTERING_TISSUE_NIR, SOUND_SPEED_TISSUE,
    SPECIFIC_HEAT_WATER,
};

/// Coupling convergence information
#[derive(Debug, Clone)]
pub struct CouplingConvergenceInfo {
    /// Whether coupling converged
    pub converged: bool,

    /// Number of Newton iterations
    pub newton_iterations: usize,

    /// Final residual norm
    pub final_residual: f64,

    /// Relative residual: ||F|| / ||F₀||
    pub relative_residual: f64,

    /// Total wall time
    pub wall_time_seconds: f64,

    /// GMRES iterations per Newton step (average)
    pub avg_gmres_iterations: usize,
}
/// Physical coefficients for the coupled acoustic-optical-thermal system
///
/// Contains material properties needed to evaluate the PDE residuals.
/// Default values correspond to soft biological tissue at 37 °C.
#[derive(Debug, Clone)]
pub struct PhysicsCoefficients {
    /// Speed of sound \[m/s\]
    pub sound_speed: f64,
    /// Mass density \[kg/m³\]
    pub density: f64,
    /// Specific heat capacity \[J/(kg·K)\]
    pub specific_heat: f64,
    /// Thermal conductivity \[W/(m·K)\]
    pub thermal_conductivity: f64,
    /// Optical absorption coefficient μ_a \[1/m\]
    pub optical_absorption: f64,
    /// Reduced scattering coefficient μ_s' \[1/m\]
    pub reduced_scattering: f64,
    /// Acoustic absorption coefficient \[Np/m\]
    pub acoustic_absorption: f64,
    /// Grüneisen parameter Γ for photoacoustic source p₀ = Γ·μₐ·Φ (dimensionless)
    ///
    /// For water at 37 °C: Γ ≈ 0.12.  Treating Γ = 1 overestimates photoacoustic
    /// amplitude by ~8×.
    ///
    /// Reference: Jacques, S.L. (1993). Appl. Opt. 32(13), 2447–2454.
    pub gruneisen: f64,
}

impl Default for PhysicsCoefficients {
    fn default() -> Self {
        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            density: DENSITY_WATER_NOMINAL,
            specific_heat: SPECIFIC_HEAT_WATER,
            thermal_conductivity: 0.6,
            optical_absorption: OPTICAL_ABSORPTION_TISSUE_NIR,
            reduced_scattering: REDUCED_SCATTERING_TISSUE_NIR,
            acoustic_absorption: ACOUSTIC_ABSORPTION_TISSUE,
            gruneisen: GRUNEISEN_WATER_37C,
        }
    }
}

impl PhysicsCoefficients {
    /// Thermal diffusivity κ = k / (ρ · cₚ)
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat)
    }

    /// Optical diffusion coefficient D = 1 / (3 · (μ_a + μ_s'))
    pub fn optical_diffusion(&self) -> f64 {
        1.0 / (3.0 * (self.optical_absorption + self.reduced_scattering))
    }
}
/// Newton-Krylov method configuration
#[derive(Debug, Clone)]
pub struct NewtonKrylovConfig {
    /// Maximum Newton iterations
    pub max_newton_iterations: usize,

    /// Newton tolerance: ||F(u)|| < tolerance
    pub newton_tolerance: f64,

    /// Line search parameter (0, 1]
    pub line_search_parameter: f64,

    /// Enable adaptive step size
    pub adaptive_step_size: bool,

    /// Verbose output
    pub verbose: bool,
}

impl Default for NewtonKrylovConfig {
    fn default() -> Self {
        Self {
            max_newton_iterations: 20,
            newton_tolerance: 1e-6,
            line_search_parameter: 1.0,
            adaptive_step_size: true,
            verbose: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_newton_krylov_config_default() {
        let config = NewtonKrylovConfig::default();
        assert_eq!(config.max_newton_iterations, 20);
        assert!(config.newton_tolerance < 1e-5);
        assert!(config.line_search_parameter > 0.0 && config.line_search_parameter <= 1.0);
    }

    #[test]
    fn test_physics_coefficients_default() {
        let c = PhysicsCoefficients::default();
        assert!((c.sound_speed - SOUND_SPEED_TISSUE).abs() < 1e-10);
        assert!(c.thermal_diffusivity() > 0.0);
        assert!(c.optical_diffusion() > 0.0);
    }

    /// Photoacoustic source term scales linearly with the Grüneisen parameter.
    #[test]
    fn test_photoacoustic_default_gruneisen_not_one() {
        let c = PhysicsCoefficients::default();
        assert!(
            (c.gruneisen - 1.0).abs() > 0.01,
            "Default Grüneisen parameter ({}) must not be 1.0; water at 37°C ≈ 0.12",
            c.gruneisen
        );
        assert!(
            c.gruneisen > 0.0,
            "Grüneisen parameter must be positive, got {}",
            c.gruneisen
        );
    }
}
