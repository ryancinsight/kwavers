//! Yoshida 4th-order symplectic composition step.

use super::{stormer_verlet_step, YOSHIDA_W1, YOSHIDA_W2};
use crate::acoustics::bubble_dynamics::bubble_state::BubbleState;
use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use kwavers_core::error::KwaversResult;

/// Yoshida 4th-order symplectic step: three Störmer-Verlet sub-steps.
///
/// # Algorithm
///
/// ```text
/// Ψ_h = Φ_{w₁h} ∘ Φ_{w₂h} ∘ Φ_{w₁h}
/// ```
/// with `w₁ ≈ 1.3512` and `w₂ ≈ −1.7024` (negative — backward sub-step).
///
/// The acoustic forcing `p·sin(ω·t)` is evaluated at the correct sub-step time
/// `t`, `t + w₁·h`, `t + (w₁+w₂)·h` respectively, so negative `w₂` works
/// correctly without any special handling.
///
/// # Arguments — same as `stormer_verlet_step`
/// # Errors
/// - Propagates any `KwaversError` returned by called functions.
///
#[allow(clippy::too_many_arguments)]
pub fn yoshida4_step(
    state: &mut BubbleState,
    model: &KellerMiksisModel,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
    max_mach: f64,
    r_min_fraction: f64,
) -> KwaversResult<()> {
    // Sub-step 1: Φ_{w₁h} starting at t
    stormer_verlet_step(
        state,
        model,
        p_acoustic,
        dp_dt,
        YOSHIDA_W1 * dt,
        t,
        max_mach,
        r_min_fraction,
    )?;

    // Sub-step 2: Φ_{w₂h} starting at t + w₁·h  (w₂ < 0 → backward step)
    let t1 = YOSHIDA_W1.mul_add(dt, t);
    stormer_verlet_step(
        state,
        model,
        p_acoustic,
        dp_dt,
        YOSHIDA_W2 * dt,
        t1,
        max_mach,
        r_min_fraction,
    )?;

    // Sub-step 3: Φ_{w₁h} starting at t + (w₁+w₂)·h
    let t2 = YOSHIDA_W2.mul_add(dt, t1);
    stormer_verlet_step(
        state,
        model,
        p_acoustic,
        dp_dt,
        YOSHIDA_W1 * dt,
        t2,
        max_mach,
        r_min_fraction,
    )?;

    Ok(())
}