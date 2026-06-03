//! Complete Epstein–Plesset gas-diffusion dissolution of a free bubble.

use super::traits::{sealed, DissolutionModel, GasDiffusionParams};
use std::f64::consts::PI;

/// Epstein–Plesset (1950) dissolution model for a free gas bubble.
///
/// ## Theorem (Epstein & Plesset 1950, *J. Chem. Phys.* 18:1505)
///
/// A gas bubble of radius `R` in a liquid with dissolved-gas saturation fraction
/// `f = C_∞/C_s`, surface tension `σ`, ambient pressure `P₀`, gas diffusivity
/// `D` and Ostwald solubility `L = C_s/ρ_g` evolves by quasi-static diffusion
/// with an unsteady boundary-layer correction:
/// ```text
///   dR/dt = −D·L · (1 − f + 2σ/(R P₀)) / (1 + 4σ/(3 R P₀)) · (1/R + 1/√(π D t))
/// ```
/// * The numerator is the concentration driving force: the interface gas
///   concentration is raised above saturation by the Laplace overpressure
///   `2σ/R` (Henry's law), so a free bubble **dissolves even in a saturated
///   liquid** (`f = 1`).
/// * The denominator `1 + 4σ/(3 R P₀)` is the curvature correction relating the
///   change in dissolved mass to the change in radius.
/// * `1/R` is the steady (large-time) flux; `1/√(π D t)` is the transient
///   diffusion-layer term that dominates early.
///
/// Setting `f = 1` and `σ = 0` gives `dR/dt = 0` (no dissolution); the
/// surface-tension term then drives the classic slow dissolution.
///
/// ## References
/// - Epstein PS, Plesset MS (1950). *J. Chem. Phys.* 18, 1505.
/// - Duncan PB, Needham D (2004). *Langmuir* 20, 2567 (experimental validation).
#[derive(Debug, Clone, Copy)]
pub struct EpsteinPlessetDissolution {
    params: GasDiffusionParams,
    /// Include the Laplace-overpressure surface-tension drive + curvature term.
    pub surface_tension: bool,
    /// Include the transient diffusion-layer term `1/√(π D t)`.
    pub transient: bool,
}

impl EpsteinPlessetDissolution {
    /// Full model: surface tension and transient term both enabled.
    #[must_use]
    pub fn new(params: GasDiffusionParams) -> Self {
        Self {
            params,
            surface_tension: true,
            transient: true,
        }
    }

    /// Quasi-static variant (no transient term) — useful for the closed-form
    /// dissolution-time comparison.
    #[must_use]
    pub fn quasi_static(params: GasDiffusionParams) -> Self {
        Self {
            params,
            surface_tension: true,
            transient: false,
        }
    }
}

impl sealed::Sealed for EpsteinPlessetDissolution {}

impl DissolutionModel for EpsteinPlessetDissolution {
    fn radius_rate(&self, radius_m: f64, time_s: f64) -> f64 {
        let r = radius_m.max(1e-15);
        let p = &self.params;
        let st = if self.surface_tension {
            2.0 * p.surface_tension / (r * p.ambient_pressure)
        } else {
            0.0
        };
        let curvature = if self.surface_tension {
            1.0 + 4.0 * p.surface_tension / (3.0 * r * p.ambient_pressure)
        } else {
            1.0
        };
        let numerator = 1.0 - p.saturation_fraction + st;
        // Transient diffusion-layer term; guarded near t = 0 where 1/√t diverges
        // (the integral ∫1/√t dt is finite — the early-time unsteady flux).
        let transient = if self.transient && time_s > 0.0 {
            1.0 / (PI * p.diffusivity * time_s).sqrt()
        } else {
            0.0
        };
        let geom = 1.0 / r + transient;
        -p.rate_scale() * (numerator / curvature) * geom
    }

    fn params(&self) -> &GasDiffusionParams {
        &self.params
    }

    fn dissolution_time(&self, r0_m: f64) -> Option<f64> {
        // Quasi-static, surface-tension-free closed form (Epstein–Plesset):
        //   dR/dt = −D·L·(1−f)/R  ⇒  R² = R₀² − 2 D L (1−f) t  ⇒  τ = R₀²/(2 D L (1−f)).
        // Valid only for an undersaturated liquid; surface tension shortens it
        // (use the integrator for the exact value).
        let undersat = 1.0 - self.params.saturation_fraction;
        if undersat > 0.0 && self.params.rate_scale() > 0.0 && r0_m > 0.0 {
            Some(r0_m * r0_m / (2.0 * self.params.rate_scale() * undersat))
        } else {
            None
        }
    }
}
