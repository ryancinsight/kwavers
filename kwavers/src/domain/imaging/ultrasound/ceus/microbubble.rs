//! `Microbubble` and `SizeDistribution` — individual microbubble physics.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};

/// Size distribution parameters
#[derive(Debug, Clone)]
pub struct SizeDistribution {
    /// Mean radius (m)
    pub mean_radius: f64,
    /// Standard deviation (m)
    pub std_dev: f64,
}

/// Individual microbubble properties
#[derive(Debug, Clone)]
pub struct Microbubble {
    /// Equilibrium radius (m)
    pub radius_eq: f64,
    /// Shell thickness (m)
    pub shell_thickness: f64,
    /// Shell elasticity (Pa)
    pub shell_elasticity: f64,
    /// Shell viscosity (Pa·s)
    pub shell_viscosity: f64,
    /// Gas polytropic index
    pub polytropic_index: f64,
    /// Surface tension (N/m)
    pub surface_tension: f64,
}

impl Microbubble {
    /// Create new microbubble with typical contrast agent properties
    pub fn new(radius: f64, shell_elasticity: f64, shell_viscosity: f64) -> Self {
        Self {
            radius_eq: radius * 1e-6,                 // Convert μm to m
            shell_thickness: radius * 1e-6 * 0.1,     // 10% of radius
            shell_elasticity: shell_elasticity * 1e3, // Convert kPa to Pa
            shell_viscosity,
            polytropic_index: 1.07, // Typical for encapsulated bubbles
            surface_tension: 0.072, // N/m for water-air interface
        }
    }

    /// Create SonoVue-like microbubble (typical clinical contrast agent)
    pub fn sono_vue() -> Self {
        Self::new(1.5, 1.0, 0.5) // 1.5 μm radius, 1 kPa elasticity, 0.5 Pa·s viscosity
    }

    /// Create Definity-like microbubble
    pub fn definit_y() -> Self {
        Self::new(2.0, 2.5, 1.0) // 2.0 μm radius, 2.5 kPa elasticity, 1.0 Pa·s viscosity
    }

    /// Compute natural resonance frequency (Hz)
    #[must_use]
    pub fn resonance_frequency(&self, _ambient_pressure: f64, _liquid_density: f64) -> f64 {
        let diameter_um = self.radius_eq * 2.0 * 1e6; // Convert to μm
        let base_freq_mhz = 6.0 / diameter_um; // MHz
        let shell_factor = 1.0 + (self.shell_elasticity / 1000.0) * 0.2; // 20% increase per kPa
        base_freq_mhz * shell_factor * 1e6 // Convert to Hz
    }

    /// Validate microbubble parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius_eq <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_eq".to_string(),
                value: self.radius_eq,
                reason: "must be positive".to_string(),
            }));
        }
        if self.shell_elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_elasticity".to_string(),
                value: self.shell_elasticity,
                reason: "must be non-negative".to_string(),
            }));
        }
        if self.shell_viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_viscosity".to_string(),
                value: self.shell_viscosity,
                reason: "must be non-negative".to_string(),
            }));
        }
        Ok(())
    }

    /// Compute acoustic scattering cross-section (m²) using the Hoff et al. (2000) model.
    ///
    /// ## Physics (Hoff, Sontum & Hovem, JASA 107(4), 2000, Eq. 7)
    ///
    /// An encapsulated bubble driven acoustically at frequency f radiates as a spherical
    /// monopole.  The scattering cross-section is:
    ///
    /// ```text
    /// σ_s(ω) = 4π R² (ω R / c_L)²  /  [(1 − Ω²)² + (δ_tot Ω)²]
    /// ```
    ///
    /// where Ω = ω/ω₀ and the dimensionless total damping is (Church 1995, JASA 97(3)):
    ///
    /// ```text
    /// δ_tot = δ_rad + δ_vis + δ_sh
    ///
    ///   δ_rad = ω₀ R / c_L                                          (radiation)
    ///   δ_vis = 4 μ_L / (ω₀ ρ_L R²)                               (liquid viscosity)
    ///   δ_sh  = 4 d_s μ_s / (ω₀ ρ_L R³)                          (shell viscosity)
    /// ```
    ///
    /// Constants: c_L = 1480 m/s, ρ_L = 1000 kg/m³, μ_L = 1.002×10⁻³ Pa·s (water, 20°C).
    #[must_use]
    pub fn scattering_cross_section(&self, frequency: f64) -> f64 {
        const C_L: f64 = 1480.0; // longitudinal speed in water at 20°C [m/s]
        const RHO_L: f64 = 1000.0; // water density [kg/m³]
        const MU_L: f64 = 1.002e-3; // dynamic viscosity of water at 20°C [Pa·s]

        let r = self.radius_eq;
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let omega0 = 2.0 * std::f64::consts::PI * self.resonance_frequency(101325.0, RHO_L);

        // Dimensionless damping components (Church 1995, Eq. A3–A5)
        let delta_rad = omega0 * r / C_L;
        let delta_vis = 4.0 * MU_L / (omega0 * RHO_L * r * r);
        let delta_sh =
            4.0 * self.shell_thickness * self.shell_viscosity / (omega0 * RHO_L * r * r * r);
        let delta_tot = (delta_rad + delta_vis + delta_sh).max(1e-12);

        let big_omega = omega / omega0;
        let denom = (1.0 - big_omega * big_omega).powi(2) + (delta_tot * big_omega).powi(2);

        // σ_s = 4π R² (ωR/c_L)² / denom
        let ka = omega * r / C_L; // dimensionless acoustic size parameter
        4.0 * std::f64::consts::PI * r * r * ka * ka / denom
    }
}
