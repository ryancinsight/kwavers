//! Gas-diffusion dissolution model trait and shared parameters.
//!
//! A dissolution model returns the radius rate `dR/dt` of a gas bubble losing
//! (or gaining) gas by diffusion across its interface. The canonical model is
//! Epstein–Plesset (1950); coated-microbubble and other variants implement the
//! same [`DissolutionModel`] contract so callers (integrators, the residual-gas
//! field, the planner) stay model-agnostic.

use kwavers_core::constants::cavitation::{
    GAS_DIFFUSION_COEFFICIENT_WATER, OSTWALD_SOLUBILITY_AIR_WATER, SURFACE_TENSION_WATER,
};
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

/// Gas/liquid transport parameters governing diffusion-driven dissolution.
///
/// The dissolution rate scales with the product `D·L` of the diffusivity and
/// the Ostwald solubility; the saturation fraction `f` sets the far-field
/// driving force (`f < 1` undersaturated → dissolve, `f > 1` supersaturated →
/// grow), and surface tension always adds a Laplace-overpressure dissolution
/// drive for a free bubble.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasDiffusionParams {
    /// Gas diffusivity in the liquid `D` [m²/s].
    pub diffusivity: f64,
    /// Ostwald solubility `L = C_s/ρ_g` [-].
    pub ostwald_solubility: f64,
    /// Dissolved-gas saturation fraction `f = C_∞/C_s` [-].
    pub saturation_fraction: f64,
    /// Surface tension `σ` [N/m].
    pub surface_tension: f64,
    /// Ambient pressure `P₀` `Pa`.
    pub ambient_pressure: f64,
}

impl GasDiffusionParams {
    /// Air bubble in water at ~20 °C with the given dissolved-gas saturation
    /// fraction (`f = 1.0` saturated, `f = 0.0` fully degassed).
    #[must_use]
    pub fn air_in_water(saturation_fraction: f64) -> Self {
        Self {
            diffusivity: GAS_DIFFUSION_COEFFICIENT_WATER,
            ostwald_solubility: OSTWALD_SOLUBILITY_AIR_WATER,
            saturation_fraction,
            surface_tension: SURFACE_TENSION_WATER,
            ambient_pressure: ATMOSPHERIC_PRESSURE,
        }
    }

    /// `D·L` [m²/s] — the diffusion–solubility product that sets the rate scale.
    #[must_use]
    #[inline]
    pub fn rate_scale(&self) -> f64 {
        self.diffusivity * self.ostwald_solubility
    }
}

pub(crate) mod sealed {
    /// Seals [`super::DissolutionModel`] so external crates cannot implement it
    /// and bypass the diffusion-physics contract (per the architecture mandate).
    pub trait Sealed {}
}

/// A gas-diffusion dissolution model: returns the bubble-wall radius rate.
///
/// Sealed — implemented only by the models in this module
/// ([`super::EpsteinPlessetDissolution`], [`super::ShellPermeationDissolution`]).
pub trait DissolutionModel: sealed::Sealed {
    /// Radius rate `dR/dt` [m/s] at radius `radius_m` and elapsed time
    /// `time_s` (the latter feeds the transient diffusion-layer term).
    fn radius_rate(&self, radius_m: f64, time_s: f64) -> f64;

    /// The transport parameters this model uses.
    fn params(&self) -> &GasDiffusionParams;

    /// Closed-form dissolution time `R₀ → 0` `s` when one exists for this model
    /// (else `None`; use the numerical integrator). The quasi-static,
    /// surface-tension-free Epstein–Plesset result is
    /// `τ = R₀² / (2·D·L·(1 − f))` for `f < 1`.
    fn dissolution_time(&self, _r0_m: f64) -> Option<f64> {
        None
    }
}
