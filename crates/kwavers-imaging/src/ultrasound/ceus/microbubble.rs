//! `Microbubble` and `CeusSizeDistribution` — individual microbubble physics.

use kwavers_core::constants::cavitation::{SURFACE_TENSION_WATER, VISCOSITY_WATER};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER};
use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};

/// Size distribution parameters
#[derive(Debug, Clone)]
pub struct CeusSizeDistribution {
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
    #[must_use]
    pub fn new(radius: f64, shell_elasticity: f64, shell_viscosity: f64) -> Self {
        Self {
            radius_eq: radius * 1e-6,                 // Convert μm to m
            shell_thickness: radius * 1e-6 * 0.1,     // 10% of radius
            shell_elasticity: shell_elasticity * 1e3, // Convert kPa to Pa
            shell_viscosity,
            polytropic_index: 1.07, // Typical for encapsulated bubbles
            surface_tension: SURFACE_TENSION_WATER,
        }
    }

    /// Create SonoVue-like microbubble (typical clinical contrast agent)
    #[must_use]
    pub fn sono_vue() -> Self {
        Self::new(1.5, 1.0, 0.5) // 1.5 μm radius, 1 kPa elasticity, 0.5 Pa·s viscosity
    }

    /// Create Definity-like microbubble
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn definit_y() -> Self {
        Self::new(2.0, 2.5, 1.0) // 2.0 μm radius, 2.5 kPa elasticity, 1.0 Pa·s viscosity
    }

    /// Natural radial-pulsation resonance frequency of an encapsulated
    /// microbubble (Hz).
    ///
    /// ## Theorem (Hoff & Sontum 2000; Marmottant et al. 2005)
    ///
    /// For a thin-shell encapsulated gas bubble in an incompressible liquid,
    /// linearising the modified Rayleigh-Plesset equation about R = R₀ gives
    ///
    /// ```text
    ///         1         √( 3·γ·p₀     12·G·d      4·σ    )
    ///  f₀ = ───── ·     ( ─────── + ────────── − ─────── )
    ///       2π·R₀       (   ρ        ρ·R₀         ρ·R₀   )
    /// ```
    ///
    /// where:
    /// - γ = `self.polytropic_index`
    /// - p₀ = `ambient_pressure` (Pa)
    /// - ρ  = `liquid_density` (kg/m³)
    /// - G·d ≈ `self.shell_elasticity · self.shell_thickness` (N/m, areal shell
    ///   stiffness)
    /// - σ  = `self.surface_tension` (N/m)
    ///
    /// In the free-bubble limit (G·d → 0, σ → 0) this reduces to the classical
    /// Minnaert (1933) formula `f₀ = (1/(2πR₀))·√(3γp₀/ρ)`.  Prior to the
    /// 2026-05-21 fix this method ignored both `ambient_pressure` and
    /// `liquid_density` and returned a hardcoded `f₀ = 6 / d_μm` MHz rule of
    /// thumb (calibrated for air-in-water at STP), giving wrong values at any
    /// non-STP condition or for any shell stiffness profile the rule was not
    /// calibrated for.
    ///
    /// ## References
    /// - Minnaert M (1933). *Philos. Mag.* 16, 235–248.
    /// - Hoff L. & Sontum P. C. (2000). *Ultrasonics* 38(1–8), 113–117.
    /// - Marmottant P. et al. (2005). *J. Acoust. Soc. Am.* 118(6), 3499–3505.
    #[must_use]
    pub fn resonance_frequency(&self, ambient_pressure: f64, liquid_density: f64) -> f64 {
        if self.radius_eq <= 0.0 || liquid_density <= 0.0 {
            return 0.0;
        }
        let r0 = self.radius_eq;
        let rho = liquid_density;
        // Areal shell stiffness G·d (N/m).
        let chi = self.shell_elasticity * self.shell_thickness;
        // Hoff-Sontum thin-shell linearisation (omega^2 form):
        //   ω₀² = 3γp₀/(ρR₀²) + 12·χ/(ρR₀³) − 4σ/(ρR₀³)
        let omega_sq = (3.0 * self.polytropic_index * ambient_pressure) / (rho * r0 * r0)
            + (12.0 * chi) / (rho * r0.powi(3))
            - (4.0 * self.surface_tension) / (rho * r0.powi(3));
        if omega_sq <= 0.0 {
            return 0.0;
        }
        omega_sq.sqrt() / (TWO_PI)
    }

    /// Validate microbubble parameters
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius_eq <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_eq".to_owned(),
                value: self.radius_eq,
                reason: "must be positive".to_owned(),
            }));
        }
        if self.shell_elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_elasticity".to_owned(),
                value: self.shell_elasticity,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if self.shell_viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_viscosity".to_owned(),
                value: self.shell_viscosity,
                reason: "must be non-negative".to_owned(),
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
        let c_l = SOUND_SPEED_WATER; // longitudinal speed in water at 20°C [m/s]
        let rho_l = DENSITY_WATER_NOMINAL;
        let mu_l = VISCOSITY_WATER; // dynamic viscosity of water at 20°C [Pa·s]

        let r = self.radius_eq;
        let omega = TWO_PI * frequency;
        let omega0 = 2.0
            * std::f64::consts::PI
            * self.resonance_frequency(
                kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE,
                rho_l,
            );

        // Dimensionless damping components (Church 1995, Eq. A3–A5)
        let delta_rad = omega0 * r / c_l;
        let delta_vis = 4.0 * mu_l / (omega0 * rho_l * r * r);
        let delta_sh =
            4.0 * self.shell_thickness * self.shell_viscosity / (omega0 * rho_l * r * r * r);
        let delta_tot = (delta_rad + delta_vis + delta_sh).max(1e-12);

        let big_omega = omega / omega0;
        let denom = (delta_tot * big_omega)
            .mul_add(delta_tot * big_omega, (1.0 - big_omega * big_omega).powi(2));

        // σ_s = 4π R² (ωR/c_L)² / denom
        let ka = omega * r / c_l; // dimensionless acoustic size parameter
        FOUR_PI * r * r * ka * ka / denom
    }
}
