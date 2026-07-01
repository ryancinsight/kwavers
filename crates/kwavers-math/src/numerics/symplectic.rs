//! Symplectic integrators for separable Hamiltonian systems
//! `H(q, p) = T(p) + V(q)`.
//!
//! Symplectic integrators preserve the geometric (symplectic) structure of
//! Hamiltonian flow, so they conserve a *modified* Hamiltonian to the order of
//! the method and exhibit **no secular energy drift** over long integrations —
//! the property that matters for many-cycle conservative dynamics (e.g. the
//! conservative part of bubble oscillation over thousands of acoustic cycles,
//! where a non-symplectic integrator's energy drift would falsely indicate
//! growth or collapse).
//!
//! Both integrators take the system as two gradient closures of a *separable*
//! Hamiltonian:
//! * `dqdt(p, out)` writes `∂H/∂p = ∂T/∂p` (the drift velocity `q̇`),
//! * `dvdq(q, out)` writes `∂H/∂q = ∂V/∂q` (the kick; the force is `−∂V/∂q`).
//!
//! A `scratch` buffer of length ≥ `q.len()` is supplied so the per-step update
//! is allocation-free.
//!
//! # References
//! - Störmer (1907); Verlet (1967), *Phys. Rev.* 159, 98.
//! - Yoshida (1990), *Phys. Lett. A* 150, 262 — 4th-order composition.
//! - Hairer, Lubich & Wanner (2006), *Geometric Numerical Integration*.

/// One Störmer–Verlet (leapfrog) step for `H(q,p) = T(p) + V(q)`; 2nd-order
/// accurate and symplectic.
///
/// Advances `(q, p)` in place by the time step `h` via the symmetric
/// half-kick / full-drift / half-kick composition.
///
/// # Panics
/// Debug-asserts that `p`, `dqdt`/`dvdq` outputs, and `scratch` all match
/// `q.len()` (the closures must write exactly `q.len()` components).
pub fn stormer_verlet_step<FT, FV>(
    q: &mut [f64],
    p: &mut [f64],
    h: f64,
    dqdt: &FT,
    dvdq: &FV,
    scratch: &mut [f64],
) where
    FT: Fn(&[f64], &mut [f64]),
    FV: Fn(&[f64], &mut [f64]),
{
    let n = q.len();
    debug_assert_eq!(p.len(), n);
    debug_assert!(scratch.len() >= n);
    let half = 0.5 * h;

    // Half kick: p -= (h/2)·∂V/∂q(q)
    dvdq(q, scratch);
    for i in 0..n {
        p[i] -= half * scratch[i];
    }
    // Full drift: q += h·∂T/∂p(p)
    dqdt(p, scratch);
    for i in 0..n {
        q[i] += h * scratch[i];
    }
    // Half kick: p -= (h/2)·∂V/∂q(q)
    dvdq(q, scratch);
    for i in 0..n {
        p[i] -= half * scratch[i];
    }
}

/// Cube root of two, `2^{1/3}`, the Yoshida composition constant.
#[inline]
fn cbrt2() -> f64 {
    2.0_f64.cbrt()
}

/// One Yoshida 4th-order symplectic step for `H(q,p) = T(p) + V(q)`.
///
/// The 4th-order method is the symmetric triple composition of the 2nd-order
/// Störmer–Verlet step with sub-steps `w₁h, w₀h, w₁h`, where
/// `w₁ = 1/(2 − 2^{1/3})` and `w₀ = −2^{1/3}/(2 − 2^{1/3})` (Yoshida 1990). The
/// sub-step weights satisfy `2w₁ + w₀ = 1`, so the three sub-steps advance the
/// state by exactly `h`; the central `w₀ < 0` back-step is what cancels the
/// 3rd-order error term.
///
/// Composing the three Störmer–Verlet steps is algebraically identical to the
/// merged leapfrog form with drift weights `[w₁/2, (w₀+w₁)/2, (w₀+w₁)/2, w₁/2]`
/// (which sum to 1) and kick weights `[w₁, w₀, w₁]`.
pub fn yoshida4_step<FT, FV>(
    q: &mut [f64],
    p: &mut [f64],
    h: f64,
    dqdt: &FT,
    dvdq: &FV,
    scratch: &mut [f64],
) where
    FT: Fn(&[f64], &mut [f64]),
    FV: Fn(&[f64], &mut [f64]),
{
    let d = 2.0 - cbrt2();
    let w1 = 1.0 / d;
    let w0 = -cbrt2() / d;
    stormer_verlet_step(q, p, w1 * h, dqdt, dvdq, scratch);
    stormer_verlet_step(q, p, w0 * h, dqdt, dvdq, scratch);
    stormer_verlet_step(q, p, w1 * h, dqdt, dvdq, scratch);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Unit harmonic oscillator H = p²/2 + ω²q²/2: T = p²/2 (∂T/∂p = p),
    // V = ω²q²/2 (∂V/∂q = ω²q). Exact solution q(t)=q0 cos ωt + (p0/ω) sin ωt;
    // the total energy E = p²/2 + ω²q²/2 is exactly conserved.
    const OMEGA: f64 = 2.3; // arbitrary frequency

    fn dqdt(p: &[f64], out: &mut [f64]) {
        out[0] = p[0];
    }
    fn dvdq(q: &[f64], out: &mut [f64]) {
        out[0] = OMEGA * OMEGA * q[0];
    }
    fn energy(q: f64, p: f64) -> f64 {
        0.5 * p * p + 0.5 * OMEGA * OMEGA * q * q
    }

    /// Symplectic integrators conserve a modified Hamiltonian: the energy error
    /// is *bounded oscillation* with no secular drift, even over many periods.
    #[test]
    fn yoshida_no_secular_energy_drift_over_many_periods() {
        let (mut q, mut p, mut s) = ([1.0], [0.0], [0.0]);
        let e0 = energy(q[0], p[0]);
        let period = 2.0 * PI / OMEGA;
        let h = period / 200.0;
        let steps = (2000.0 * period / h) as usize; // 2000 periods
        let mut emax = 0.0_f64;
        for _ in 0..steps {
            yoshida4_step(&mut q, &mut p, h, &dqdt, &dvdq, &mut s);
            emax = emax.max((energy(q[0], p[0]) - e0).abs() / e0);
        }
        // Bounded relative energy error, no growth to O(1) over 2000 periods.
        assert!(emax < 1e-6, "Yoshida max relative energy error {emax}");
        // End energy ≈ start energy (no drift): the error did not accumulate.
        let e_end = (energy(q[0], p[0]) - e0).abs() / e0;
        assert!(e_end < 1e-6, "Yoshida end energy drift {e_end}");
    }

    /// One-period phase-space error vs the exact solution (start state recurs
    /// after a full period), for `n` steps with the given integrator.
    fn one_period_error(use_yoshida: bool, n: usize) -> f64 {
        let period = 2.0 * PI / OMEGA;
        let (mut q, mut p, mut s) = ([1.0_f64], [0.0_f64], [0.0_f64]);
        let h = period / n as f64;
        for _ in 0..n {
            if use_yoshida {
                yoshida4_step(&mut q, &mut p, h, &dqdt, &dvdq, &mut s);
            } else {
                stormer_verlet_step(&mut q, &mut p, h, &dqdt, &dvdq, &mut s);
            }
        }
        ((q[0] - 1.0).powi(2) + p[0].powi(2)).sqrt()
    }

    /// Störmer–Verlet is 2nd order, Yoshida is 4th order: halving the step
    /// reduces the one-period phase-space error by ≈4× and ≈16× respectively.
    #[test]
    fn convergence_orders_are_two_and_four() {
        let sv_ratio = one_period_error(false, 100) / one_period_error(false, 200);
        let yo_ratio = one_period_error(true, 100) / one_period_error(true, 200);
        assert!(
            (3.0..=5.0).contains(&sv_ratio),
            "Störmer–Verlet ratio {sv_ratio}"
        );
        assert!(
            (12.0..=20.0).contains(&yo_ratio),
            "Yoshida ratio {yo_ratio}"
        );
        // Yoshida is far more accurate than Störmer–Verlet at the same step.
        assert!(one_period_error(true, 100) < one_period_error(false, 100));
    }

    /// Yoshida sub-step weights advance time by exactly h (2w₁ + w₀ = 1) — the
    /// property the chapter's draft coefficients violated (they summed to −0.35).
    #[test]
    fn yoshida_substep_weights_sum_to_one() {
        let d = 2.0 - 2.0_f64.cbrt();
        let w1 = 1.0 / d;
        let w0 = -2.0_f64.cbrt() / d;
        assert!(
            (2.0 * w1 + w0 - 1.0).abs() < 1e-15,
            "2w1+w0 = {}",
            2.0 * w1 + w0
        );
        // Merged leapfrog drift weights also sum to 1.
        let drifts = [w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0];
        assert!((drifts.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }
}
