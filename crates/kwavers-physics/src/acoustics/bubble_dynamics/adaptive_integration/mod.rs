//! Adaptive time-stepping for bubble dynamics
//!
//! This module provides adaptive integration methods specifically designed
//! for the stiff ODEs in bubble dynamics, particularly during violent collapse.
//!
//! Two paths share the [`AdaptiveBubbleModel`] contract and the
//! step-doubling controller shape: the explicit RK4
//! [`AdaptiveBubbleIntegrator`] for weakly driven regimes, and the
//! implicit SDIRK2 [`ImplicitBubbleIntegrator`] for inertial-collapse
//! trajectories, whose collapse spike the explicit path cannot
//! traverse without injecting energy (KW-PHYS-009 item 5).
//!
//! ## Literature References
//!
//! 1. **Hairer & Wanner (1996)**. "Solving Ordinary Differential Equations II:
//!    Stiff and Differential-Algebraic Problems"
//!    - Adaptive time-stepping strategies for stiff ODEs
//!
//! 2. **Storey & Szeri (2000)**. "Water vapour, sonoluminescence and sonochemistry"
//!    - Time scales in bubble dynamics
//!
//! 3. **Lauterborn & Kurz (2010)**. "Physics of bubble oscillations"
//!    - Numerical challenges in bubble dynamics

mod config;
mod implicit;
mod integrator;
mod statistics;

#[cfg(test)]
mod tests;

pub use config::AdaptiveBubbleConfig;
pub use implicit::ImplicitBubbleIntegrator;
pub use integrator::AdaptiveBubbleIntegrator;
pub use statistics::IntegrationStatistics;

use super::{
    acoustic_pressure::InstantaneousPressure, bubble_state::BubbleParameters, BubbleState,
};
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::KwaversResult;

/// Implement [`AdaptiveBubbleModel`] for a wrapper type that holds an inner
/// field implementing the trait, by forwarding all four methods.
///
/// # Usage
///
/// ```rust,ignore
/// delegate_adaptive_bubble_model!(KellerHerringModel, inner);
/// ```
///
/// This expands to a full `impl AdaptiveBubbleModel for KellerHerringModel`
/// that delegates every method call to `self.inner`.
#[macro_export]
macro_rules! delegate_adaptive_bubble_model {
    ($ty:ty, $field:ident) => {
        impl $crate::acoustics::bubble_dynamics::adaptive_integration::AdaptiveBubbleModel for $ty {
            fn params(
                &self,
            ) -> &$crate::acoustics::bubble_dynamics::bubble_state::BubbleParameters {
                self.$field.params()
            }

            fn calculate_acceleration(
                &self,
                state: &mut $crate::acoustics::bubble_dynamics::BubbleState,
                p_acoustic: $crate::acoustics::bubble_dynamics::InstantaneousPressure,
                dp_dt: aequitas::systems::si::quantities::PressureRate,
                t: f64,
            ) -> kwavers_core::error::KwaversResult<f64> {
                self.$field
                    .calculate_acceleration(state, p_acoustic, dp_dt, t)
            }

            fn update_temperature(
                &self,
                state: &mut $crate::acoustics::bubble_dynamics::BubbleState,
                dt: f64,
            ) -> kwavers_core::error::KwaversResult<()> {
                self.$field.update_temperature(state, dt)
            }

            fn update_mass_transfer(
                &self,
                state: &mut $crate::acoustics::bubble_dynamics::BubbleState,
                dt: f64,
            ) -> kwavers_core::error::KwaversResult<()> {
                self.$field.update_mass_transfer(state, dt)
            }
        }
    };
}

/// Contract implemented by bubble-equation models used by the adaptive RK4
/// engine in this module.
pub trait AdaptiveBubbleModel {
    /// Shared model parameters (sound speed, density, equilibrium radius, etc.).
    fn params(&self) -> &BubbleParameters;

    /// Bubble-wall acceleration R̈ at the current state.
    ///
    /// # Acoustic-drive contract
    ///
    /// `p_acoustic` is the INSTANTANEOUS acoustic pressure `p_∞(t) − p0`
    /// — typed as [`InstantaneousPressure`], so an amplitude cannot be
    /// passed here by mistake — and `dp_dt` is its time derivative
    /// `dp_∞/dt` as a [`PressureRate`]. The bubble equations are written
    /// in terms of a general far-field pressure `p_∞(t)` — Keller &
    /// Miksis (1980, JASA 68(2):628–633), Gilmore (1952, Caltech Hydro.
    /// Lab. Report 26-4); the sinusoidal form
    /// `p_∞(t) = p0 ± P_A·sin(ωt + φ)` is the standard specialisation
    /// for continuous harmonic drive, not part of the equations. Callers
    /// that have an amplitude synthesise the tone themselves with
    /// [`PressureAmplitude::sample`](crate::acoustics::bubble_dynamics::acoustic_pressure::PressureAmplitude::sample).
    /// The `t` argument is retained for
    /// interface compatibility; the drive is fully specified by
    /// (`p_acoustic`, `dp_dt`).
    ///
    /// The integrator holds (`p_acoustic`, `dp_dt`) constant across one
    /// main step (zero-order hold) and varies only `t` across its RK4
    /// stages, so callers must keep the main step small against the
    /// acoustic period (≲ T/200; the triplet-pulse port uses T/500, the
    /// passive-dose simulator caps at T/200).
    ///
    /// # Errors
    ///
    /// Returns a model-specific error when the state or acoustic inputs violate the equation's
    /// domain.
    fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: InstantaneousPressure,
        dp_dt: PressureRate,
        t: f64,
    ) -> KwaversResult<f64>;

    /// Optional temperature update hook. A no-op implementation is valid when
    /// thermal effects are disabled for the selected model.
    ///
    /// # Errors
    ///
    /// Propagates a model-specific thermodynamic update failure.
    fn update_temperature(&self, state: &mut BubbleState, dt: f64) -> KwaversResult<()>;

    /// Optional vapor transfer update hook. A no-op implementation is valid when
    /// mass transfer is disabled for the selected model.
    ///
    /// # Errors
    ///
    /// Propagates a model-specific mass-transfer update failure.
    fn update_mass_transfer(&self, state: &mut BubbleState, dt: f64) -> KwaversResult<()>;
}

/// Replace the old fixed-timestep integration with adaptive version.
///
/// `p_acoustic` is the INSTANTANEOUS acoustic pressure `p_∞(t) − p0` at
/// the step start and `dp_dt` its time derivative; both are held constant
/// across the step's sub-cycling (zero-order hold — see the contract on
/// [`AdaptiveBubbleModel::calculate_acceleration`]). Keep `dt` small
/// against the acoustic period (≲ T/200).
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn integrate_bubble_dynamics_adaptive<Model>(
    solver: &Model,
    state: &mut BubbleState,
    p_acoustic: InstantaneousPressure,
    dp_dt: PressureRate,
    dt: f64,
    t: f64,
) -> KwaversResult<()>
where
    Model: AdaptiveBubbleModel,
{
    let config = AdaptiveBubbleConfig::default();
    let mut integrator = AdaptiveBubbleIntegrator::new(solver, config);
    integrator.integrate_adaptive(state, p_acoustic, dp_dt, dt, t)
}

/// Integrate one main step with the implicit SDIRK2 path
/// ([`ImplicitBubbleIntegrator`]) — the stiff-capable counterpart of
/// [`integrate_bubble_dynamics_adaptive`], for trajectories that
/// collapse inertially (KW-PHYS-009 item 5). Same zero-order-hold
/// drive contract as the explicit path.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn integrate_bubble_dynamics_implicit<Model>(
    solver: &Model,
    state: &mut BubbleState,
    p_acoustic: InstantaneousPressure,
    dp_dt: PressureRate,
    dt: f64,
    t: f64,
) -> KwaversResult<()>
where
    Model: AdaptiveBubbleModel,
{
    let config = AdaptiveBubbleConfig::default();
    let mut integrator = ImplicitBubbleIntegrator::new(solver, config);
    integrator.integrate_adaptive(state, p_acoustic, dp_dt, dt, t)
}
