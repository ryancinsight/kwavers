//! Bubble Dynamics Integration — Symplectic and Geometric Methods
//!
//! ## Motivation
//!
//! Long-time bubble dynamics (sonoluminescence, multi-cycle HIFU, bubble clouds)
//! require energy-preserving integrators.  Forward Euler has first-order energy
//! drift that grows linearly with simulation time.  The Störmer-Verlet integrator
//! employed here is *symplectic*: it exactly preserves a modified Hamiltonian
//! H̃ = H + O(h²), bounding the energy error for all time.
//!
//! ## Integrator Selection
//!
//! | Use case | Integrator |
//! |---|---|
//! | Undriven, long-time (`integrate_bubble_dynamics_stable`) | Störmer-Verlet (h² global error, symplectic) |
//! | Driven + thermal coupling | IMEX Euler in `physics::acoustics::bubble_dynamics::imex_integration` |
//! | Highest accuracy, long-time | Yoshida 4th-order in `solver::forward::ode::bubble_symplectic` |
//!
//! ## Theorem — Störmer-Verlet Symplecticity
//!
//! Let `(R, V)` be the generalised coordinate and momentum of the Rayleigh-Plesset
//! mechanical sub-system with Hamiltonian `H(R, V) = ½ρ_L R³ V² + V_eff(R)`.
//! The Störmer-Verlet map Φ_h preserves the symplectic 2-form ω = dR ∧ dV:
//!
//! ```text
//!   V_{n+½} = V_n   + (h/2) f(R_n,  V_n)
//!   R_{n+1} = R_n   + h     V_{n+½}
//!   V_{n+1} = V_{n+½} + (h/2) f(R_{n+1}, V_{n+½})
//! ```
//!
//! **Proof sketch**: ∂(R_{n+1}, V_{n+1})/∂(R_n, V_n) is symplectic — its
//! transpose-inverse equals itself w.r.t. J = [[0,1],[-1,0]].
//!
//! **Consequence**: ∃ modified Hamiltonian H̃ = H + h²H₂ + O(h⁴) that is exactly
//! conserved, so `|H(t) − H(0)|` remains bounded (no secular drift).
//!
//! ## References
//!
//! - Hairer E, Lubich C, Wanner G (2006). *Geometric Numerical Integration*, 2nd ed.
//!   Springer. Theorem VI.2.2. DOI:10.1007/3-540-30666-8
//! - Yoshida H (1990). Phys. Lett. A **150**(5-7):262–268.
//! - Brennen CE (1995). *Cavitation and Bubble Dynamics*. Oxford. §2.3.

use std::sync::Arc;

use super::bubble_state::{BubbleParameters, BubbleState};
use super::keller_miksis::KellerMiksisModel;
use crate::core::error::{KwaversError, KwaversResult, PhysicsError};
use crate::solver::forward::ode::bubble_symplectic::{BubbleSymplecticIntegrator, SymplecticConfig};

/// Integrate bubble dynamics using the Störmer-Verlet symplectic integrator.
///
/// ## Algorithm
///
/// The undriven (or weakly driven) K-M mechanical system is integrated using
/// the Störmer-Verlet method, which is 2nd-order accurate and exactly symplectic.
/// Acoustic driving is set to zero (`p_acoustic = 0`, `dp_dt = 0`) — for a driven
/// bubble use `BubbleSymplecticIntegrator::sv_step` directly.
///
/// The integration advances from `time_span.0` to `time_span.1` using fixed
/// time step `dt`.  If the bubble collapses (`R < r_min`) the integration
/// terminates early and returns the clamped state.
///
/// ## Stability
///
/// The Störmer-Verlet method is unconditionally stable for linear oscillators and
/// conditionally stable for nonlinear systems.  Stability requires:
/// ```text
///   dt ≤ T_Minnaert / (2π·CFL)    where CFL ≈ 10 for typical bubble parameters
/// ```
///
/// ## References
/// - Hairer et al. (2006) §VI.2. Theorem VI.2.2.
/// - Minnaert M (1933). Phil. Mag. 16:235–248. (natural frequency)
pub fn integrate_bubble_dynamics_stable(
    initial_state: BubbleState,
    params: &BubbleParameters,
    time_span: (f64, f64),
    dt: f64,
) -> KwaversResult<BubbleState> {
    if dt <= 0.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "dt".to_string(),
            value: dt,
            reason: "Time step must be positive".to_string(),
        }));
    }
    if time_span.1 <= time_span.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "time_span".to_string(),
            value: time_span.1,
            reason: "End time must be greater than start time".to_string(),
        }));
    }

    let model = Arc::new(KellerMiksisModel::new(params.clone()));
    let config = SymplecticConfig {
        dt,
        ..SymplecticConfig::default()
    };
    let integrator = BubbleSymplecticIntegrator::new(model, config);

    let mut state = initial_state;
    let mut t = time_span.0;

    // Fixed-step Störmer-Verlet loop.  The final step may advance t slightly
    // past time_span.1 by at most dt — acceptable for physics integration where
    // the exact end time is not critical.
    while t < time_span.1 {
        match integrator.sv_step(&mut state, 0.0, 0.0, t) {
            Ok(()) => {}
            Err(KwaversError::Physics(PhysicsError::NumericalInstability { .. })) => {
                // Mach number exceeded — bubble has collapsed; return clamped state.
                return Ok(state);
            }
            Err(e) => return Err(e),
        }
        t += dt;
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_input_validation() {
        let params = BubbleParameters::default();
        let state = BubbleState::new(&params);

        let result = integrate_bubble_dynamics_stable(state.clone(), &params, (0.0, 1.0), -0.1);
        assert!(result.is_err(), "negative dt must error");

        let result = integrate_bubble_dynamics_stable(state.clone(), &params, (1.0, 0.0), 0.1);
        assert!(result.is_err(), "reversed time span must error");
    }

    #[test]
    fn test_integration_returns_valid_state() {
        let params = BubbleParameters::default();
        let state = BubbleState::new(&params);

        let result = integrate_bubble_dynamics_stable(state, &params, (0.0, 1e-6), 1e-9);
        assert!(result.is_ok());
        let final_state = result.unwrap();
        assert!(final_state.radius > 0.0, "radius must remain positive");
    }

    /// At exact equilibrium (R=R₀, Ṙ=0) the Störmer-Verlet integration
    /// must keep R within 1% of R₀ over 100 steps (no spurious drift).
    #[test]
    fn test_equilibrium_stable_for_100_steps() {
        let params = BubbleParameters::default();
        let state = BubbleState::at_equilibrium(&params);
        let r0 = params.r0;
        let dt = 1e-10; // ~ 0.01 × Minnaert period for R₀ = 1 µm

        let result = integrate_bubble_dynamics_stable(state, &params, (0.0, 100.0 * dt), dt);
        let final_state = result.unwrap();
        let rel_err = (final_state.radius - r0).abs() / r0;
        assert!(
            rel_err < 0.01,
            "equilibrium radius drift: |R−R₀|/R₀ = {rel_err:.3e} > 1%"
        );
    }
}
