//! Canonical encapsulated-bubble shell-model trait.
//!
//! Every shelled-microbubble model in the literature (Church 1995, de Jong 1994,
//! Hoff 2000, Marmottant 2005, Sarkar 2005) is the same modified Rayleigh-Plesset
//! balance
//!
//! ```text
//! ρ (R R̈ + 3/2 Ṙ²) = p_gas − p_∞ − 2 σ_eff(R)/R − 4 μ_L Ṙ/R − S_shell(R, Ṙ)
//! ```
//!
//! and differs only in three model-specific pieces:
//! - the equilibrium gas-pressure reference `p_eq` (gas obeys `p_gas =
//!   p_eq·(R0/R)^{3γ}`),
//! - the effective interfacial tension `σ_eff(R)` entering the `2σ/R` term
//!   (constant for Church/de Jong/Hoff, radius-dependent for Marmottant/Sarkar),
//! - the shell stress `S_shell(R, Ṙ)` (elastic restoring + viscous damping).
//!
//! [`EncapsulatedShellModel`] captures exactly those three as required methods and
//! provides the Rayleigh-Plesset time-derivative once, as the single source of
//! truth, so the individual models carry no duplicated RP arithmetic.

use crate::acoustics::bubble_dynamics::bubble_state::{
    young_laplace_pressure, BubbleParameters, BubbleState,
};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;

/// An encapsulated (shelled) microbubble dynamics model over the modified
/// Rayleigh-Plesset equation. See the [module docs](self) for the shared balance.
pub trait EncapsulatedShellModel {
    /// Bubble/liquid parameters (ambient pressure, density, viscosity, R0, …).
    fn params(&self) -> &BubbleParameters;

    /// Equilibrium gas-pressure reference `p_eq` [Pa]; the internal gas obeys
    /// `p_gas = p_eq·(R0/R)^{3γ}`. Models set this so that the static balance at
    /// `R = R0` matches their reference interfacial tension.
    fn equilibrium_gas_pressure(&self) -> f64;

    /// Effective interfacial tension `σ_eff(R)` [N/m] entering the `2σ_eff/R`
    /// term. Constant for elastic-shell-stress models; radius-dependent for
    /// models (Marmottant, Sarkar) that fold shell elasticity into the tension.
    fn effective_surface_tension(&self, r: f64) -> f64;

    /// Shell-contributed stress `S_shell(R, Ṙ)` [Pa] (elastic restoring + viscous
    /// damping) beyond the standard `2σ/R + 4μ_L Ṙ/R`. Positive resists expansion.
    fn shell_stress(&self, r: f64, v: f64) -> f64;

    /// Bubble-wall acceleration `R̈` from the modified Rayleigh-Plesset equation.
    ///
    /// Single source of truth for the RP time-derivative shared by every shell
    /// model. Updates `state.{wall_acceleration, pressure_internal, pressure_liquid}`.
    /// The acoustic forcing is `p_∞ = p0 + p_acoustic·sin(ωt)`.
    ///
    /// # Errors
    /// Returns a model error if a sub-step fails; the base RP path is infallible.
    fn acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let p = self.params();
        let r = state.radius;
        let v = state.wall_velocity;

        // Acoustic forcing (kept as p0 + amp·sin(ωt) — not fused — so refactored
        // models are bit-identical to their original inline implementations).
        let omega = TWO_PI * p.driving_frequency;
        let p_inf = p.p0 + p_acoustic * (omega * t).sin();

        let gamma = state.gas_species.gamma();
        let p_gas = self.equilibrium_gas_pressure() * (p.r0 / r).powf(3.0 * gamma);

        let surface_term = young_laplace_pressure(self.effective_surface_tension(r), r);
        let viscous_liquid = p.viscous_wall_stress(v, r);
        let shell = self.shell_stress(r, v);

        let net_pressure = p_gas - p_inf - surface_term - viscous_liquid - shell;
        let accel = (net_pressure / (p.rho_liquid * r)) - (1.5 * v * v / r);

        state.wall_acceleration = accel;
        state.pressure_internal = p_gas;
        state.pressure_liquid = p_inf;
        Ok(accel)
    }
}
