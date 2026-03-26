//! Core stability solver implementing the Epstein-Plesset theorem
//!
//! ## Mathematical Foundation
//!
//! Starting from the Rayleigh-Plesset equation:
//! ```text
//! R̈ + (3/2)Ṙ² = (1/ρ)(p_gas - p∞ - 2σ/R - 4μṘ/R)
//! ```
//!
//! For small oscillations around equilibrium (R = R₀ + δR), linearization gives:
//! ```text
//! δR̈ + δω²δR = 0
//! ```
//!
//! ## References
//! - Epstein, P. S., & Plesset, M. S. (1953). J. Chem. Phys., 18(11), 1505-1509.
//! - Prosperetti, A. (1984). Appl. Sci. Res., 38(3), 145-164.

use super::types::{OscillationType, StabilityAnalysis};
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;

/// Epstein-Plesset stability solver
#[derive(Debug)]
pub struct EpsteinPlessetStabilitySolver {
    /// Bubble parameters
    pub(crate) params: BubbleParameters,
}

impl EpsteinPlessetStabilitySolver {
    /// Create new Epstein-Plesset stability solver
    #[must_use]
    pub fn new(params: BubbleParameters) -> Self {
        Self { params }
    }

    /// Perform complete stability analysis for bubble oscillations
    #[must_use]
    pub fn analyze_stability(&self) -> StabilityAnalysis {
        let stability_param = self.compute_stability_parameter();
        let damping = self.compute_damping_coefficient();
        let resonance_freq = self.compute_resonance_frequency();
        let quality_factor = resonance_freq / (2.0 * damping);

        let oscillation_type = if stability_param > 0.0 {
            OscillationType::StableHarmonic
        } else if stability_param == 0.0 {
            OscillationType::Marginal
        } else {
            OscillationType::Unstable
        };

        let is_stable = stability_param >= 0.0;

        StabilityAnalysis {
            resonance_frequency: resonance_freq,
            stability_parameter: stability_param,
            damping_coefficient: damping,
            quality_factor,
            is_stable,
            oscillation_type,
        }
    }

    /// Compute the stability parameter δω² from Epstein-Plesset analysis
    ///
    /// δω² = ω₀²(γ-1)/γ + 2σ/(ρR₀³) - viscous terms
    #[must_use]
    pub(crate) fn compute_stability_parameter(&self) -> f64 {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let sigma = self.params.sigma;
        let mu = self.params.mu_liquid;
        let gamma = self.params.gamma;

        let omega0_squared = (3.0 * gamma * self.params.p0) / (rho * r0 * r0);
        let surface_term = (2.0 * sigma) / (rho * r0 * r0 * r0);
        let viscous_term = (4.0 * mu * mu) / (rho * rho * r0 * r0 * r0 * r0);

        let gas_term = if self.params.use_thermal_effects {
            omega0_squared * (gamma - 1.0) / gamma
        } else {
            0.0
        };

        gas_term + surface_term - viscous_term
    }

    /// Compute viscous damping coefficient [1/s]
    ///
    /// δ(t) = 2μ/(ρR₀²) for small oscillations
    #[must_use]
    pub(crate) fn compute_damping_coefficient(&self) -> f64 {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let mu = self.params.mu_liquid;
        2.0 * mu / (rho * r0 * r0)
    }

    /// Compute resonance frequency using Minnaert formula [Hz]
    ///
    /// f₀ = (1/2πR₀)√(3γP₀/ρ)
    #[must_use]
    pub(crate) fn compute_resonance_frequency(&self) -> f64 {
        let r0 = self.params.r0;
        let rho = self.params.rho_liquid;
        let gamma = self.params.gamma;
        let omega0 = ((3.0 * gamma * self.params.p0) / (rho * r0 * r0)).sqrt();
        omega0 / (2.0 * std::f64::consts::PI)
    }
}
