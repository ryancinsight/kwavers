//! Symplectic Integration for Bubble Dynamics
//!
//! Implements geometric numerical integrators that preserve the Hamiltonian
//! structure of the Keller-Miksis / Rayleigh-Plesset mechanical sub-system.
//!
//! # Mathematical Foundation
//!
//! ## Hamiltonian Structure
//!
//! The undamped, undriven bubble oscillation is a Hamiltonian system with
//! coordinate `R` (radius) and wall velocity `Ṙ`. Per unit solid angle (i.e.
//! dividing the liquid energy `2πρ_L R³Ṙ² − ∫p_net·4πR'²dR'` by `4π`):
//! ```text
//! H(R, Ṙ) = ½ ρ_L R³ Ṙ²  +  V_eff(R)
//! ```
//! where `V_eff(R) = −∫ p_net(R′) R′² dR′` and `p_net = p_gas − p₀ − 2σ/R`. The
//! `4π` factor is omitted consistently from both terms, so it cancels in the
//! dimensionless energy-drift ratio `H(t)/H(0)`.
//!
//! The equations of motion are `dR/dt = Ṙ` and `d(Ṙ)/dt = f(R, Ṙ)` where
//! `f = R̈` is evaluated by the Keller-Miksis RHS (see `equation.rs`).
//!
//! ## Störmer-Verlet (velocity-Verlet) integrator
//!
//! The map Φ_h: (R_n, Ṙ_n) → (R_{n+1}, Ṙ_{n+1}) defined by
//! ```text
//! V_{n+½} = V_n   + (h/2) f(R_n, V_n)
//! R_{n+1} = R_n   + h · V_{n+½}
//! V_{n+1} = V_{n+½} + (h/2) f(R_{n+1}, V_{n+½})
//! ```
//! is **2nd-order accurate** (global error O(h²)).
//!
//! **It is NOT canonically symplectic for this system.** Velocity-Verlet
//! preserves `ω = dR ∧ dṘ` only when the force is *position-only*, `f = f(R)`
//! (separable `H = T(p) + V(q)`); the textbook `det J = 1` argument relies on
//! `∂f/∂Ṙ = 0`. The RP wall acceleration is **velocity-dependent**,
//! `f = (p_B−p_∞)/(ρ_L R) − (3/2)Ṙ²/R`, whose geometric term `−3Ṙ²/(2R)` reflects
//! the position-dependent effective mass `M(R) ∝ R³` (so the canonical momentum is
//! `p = M(R)Ṙ`, not `Ṙ`), and Keller-Miksis adds velocity-dependent radiation
//! damping. Consequently no modified Hamiltonian is conserved to O(h²): the energy
//! oscillation is O(h), measured at ≈17 % over 1000 conservative periods at
//! `h = T/200` (vs the ≪1 % of a separable symplectic scheme). The energy is
//! nonetheless **bounded** (no secular drift — it stays in `[0.5 H₀, H₀]`), which
//! with 2nd-order accuracy is what this integrator actually guarantees. A genuinely
//! separable symplectic integrator lives in `kwavers_math::numerics::symplectic`.
//!
//! Reference: Hairer, E., Lubich, C. & Wanner, G. (2006).
//! *Geometric Numerical Integration*, 2nd ed. Springer. §VI (velocity-Verlet is
//! symplectic only for separable Hamiltonians).
//!
//! ## Yoshida 4th-Order Composition
//!
//! **Theorem (Yoshida 1990, eq. 2.11).**
//! Let Φ_h be any symmetric 2nd-order integrator. The triple composition
//! ```text
//! Ψ_h = Φ_{w₁h} ∘ Φ_{w₂h} ∘ Φ_{w₁h}
//! ```
//! with coefficients
//! ```text
//! w₁ =  1 / (2 − 2^{1/3}) ≈  1.3512071919596577
//! w₂ = −2^{1/3} / (2 − 2^{1/3}) ≈ −1.7024143839193153
//! ```
//! raises the order of `Φ_h` from 2 to 4 (global error O(h⁴)); the result is
//! symplectic *iff* the base `Φ_h` is. Here `Φ_h` is the velocity-Verlet step
//! above, which is symplectic only for a position-only force — so for the
//! velocity-dependent RP/Keller-Miksis force this composition is **4th-order
//! accurate but not canonically symplectic** (bounded, not modified-Hamiltonian-
//! conserved, long-time energy).
//! **Order proof:** Baker-Campbell-Hausdorff. Consistency: 2w₁ + w₂ = 1.
//! Leading error cancellation: 2w₁³ + w₂³ = 0. These two equations have the
//! unique solution above. Global error: O(h⁴).
//!
//! Note: the middle sub-step has w₂ < 0 (backward-in-time). This is valid as
//! long as the acoustic forcing p_acoustic·sin(ω(t+w₂h)) is evaluated at the
//! correct sub-step time.
//!
//! Reference: Yoshida, H. (1990). *Phys. Lett. A* **150**(5-7), 262–268.
//!
//! ## Comparison with Existing Integrators
//!
//! | Integrator | Order | Canonically symplectic? | Long-time energy |
//! |------------|-------|------------|-------------------|
//! | Forward Euler (`integration.rs`) | 1 | No | Secular drift O(h)/step |
//! | SSP-IMEX ERK2 (`bubble_imex.rs`) | 2 | No | Bounded for stiff thermal |
//! | **Störmer-Verlet** (this file) | 2 | No¹ | Bounded, O(h) oscillation |
//! | **Yoshida4** (this file) | 4 | No¹ | Bounded, smaller oscillation |
//!
//! ¹ Velocity-Verlet/Yoshida are symplectic only for a position-only force. The
//!   RP/Keller-Miksis wall acceleration is velocity-dependent, so these are
//!   2nd/4th-order accurate with *bounded* (not modified-Hamiltonian-conserved)
//!   long-time energy. See `kwavers_math::numerics::symplectic` for a genuinely
//!   symplectic integrator on a separable Hamiltonian.
//!
//! # Module layout
//!
//! - `stormer_verlet`: 2nd-order symplectic Störmer-Verlet step kernel.
//! - `yoshida`: 4th-order Yoshida composition kernel.
//! - `integrate`: convenience time-span integration wrapper.
//!
//! # Validation
//!
//! - `test_minnaert_period`: Minnaert frequency error < 0.5% at dt = T₀/200
//! - `test_hamiltonian_no_drift`: H stays in [0.5 H₀, 2 H₀] over 1000 periods
//! - `test_yoshida4_order`: Convergence order 4.0 ± 30% on SHO
//! - `test_equilibrium_preserved`: |R−R₀|/R₀ < 1e-12 at exact equilibrium

mod integrate;
mod stormer_verlet;
mod yoshida;

#[cfg(test)]
mod tests;

pub use integrate::integrate_bubble_dynamics_symplectic;
pub use stormer_verlet::stormer_verlet_step;
pub use yoshida::yoshida4_step;

use std::sync::Arc;

use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use kwavers_core::error::KwaversResult;

// ─── Yoshida composition coefficients ─────────────────────────────────────────
//
// w₁ = 1/(2 − 2^{1/3}),  w₂ = −2^{1/3}/(2 − 2^{1/3})
// Satisfy: 2w₁ + w₂ = 1 (consistency),  2w₁³ + w₂³ = 0 (4th-order condition)
pub(super) const YOSHIDA_W1: f64 = 1.351_207_191_959_657_7;
pub(super) const YOSHIDA_W2: f64 = -1.702_414_383_919_315_3;

/// Configuration for symplectic bubble integrators.
#[derive(Debug, Clone, PartialEq)]
pub struct SymplecticConfig {
    /// Default time step (s).
    pub dt: f64,
    /// Maximum allowed wall Mach number before returning an error.
    /// The K-M equation is singular at M = 1; 0.95 is a conservative bound.
    pub max_mach: f64,
    /// Minimum radius as a fraction of R₀ (hard collapse floor).
    /// Prevents division-by-zero during violent collapse.
    pub r_min_fraction: f64,
}

impl Default for SymplecticConfig {
    fn default() -> Self {
        Self {
            dt: 1e-9,
            max_mach: 0.95,
            r_min_fraction: 0.01,
        }
    }
}

/// Public wrapper struct providing the Störmer-Verlet and Yoshida4 integrators
/// as a stateful object (holds model and config).
#[derive(Debug)]
pub struct BubbleSymplecticIntegrator {
    model: Arc<KellerMiksisModel>,
    pub config: SymplecticConfig,
}

impl BubbleSymplecticIntegrator {
    /// Create integrator with default config.
    #[must_use]
    pub fn with_defaults(model: Arc<KellerMiksisModel>) -> Self {
        Self {
            model,
            config: SymplecticConfig::default(),
        }
    }

    /// Create integrator with explicit config.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(model: Arc<KellerMiksisModel>, config: SymplecticConfig) -> Self {
        Self { model, config }
    }

    /// Advance state by one Störmer-Verlet step (2nd-order symplectic).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn sv_step(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        stormer_verlet_step(
            state,
            &self.model,
            p_acoustic,
            dp_dt,
            self.config.dt,
            t,
            self.config.max_mach,
            self.config.r_min_fraction,
        )
    }

    /// Advance state by one Yoshida 4th-order step.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn yoshida4_step(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        yoshida4_step(
            state,
            &self.model,
            p_acoustic,
            dp_dt,
            self.config.dt,
            t,
            self.config.max_mach,
            self.config.r_min_fraction,
        )
    }
}
