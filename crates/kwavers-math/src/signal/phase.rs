//! Canonical phase-angle utilities (single source of truth).
//!
//! Phase differences must be compared on the circle, where `θ` and `θ ± 2πk`
//! are identical. Several call sites (the instantaneous-phase FWI misfit and the
//! PINN/training phase losses) independently re-implemented the wrap to the
//! principal interval; this module is the SSOT for that operation.

use kwavers_core::constants::numerical::TWO_PI;

/// Wrap a phase angle to the principal interval `(−π, π]`.
///
/// ## Theorem
/// For any finite `θ`, `wrap_to_pi(θ) ≡ θ (mod 2π)` and
/// `wrap_to_pi(θ) ∈ (−π, π]`. The map is the unique representative of the
/// equivalence class `{θ + 2πk : k ∈ ℤ}` in that half-open interval, so it is
/// the minimal-magnitude signed phase difference.
///
/// ## Implementation
/// `r = θ.rem_euclid(2π) ∈ [0, 2π)`; subtract `2π` when `r > π` to fold the
/// upper half into `(−π, 0)`. `rem_euclid` is branch-free and exact for finite
/// inputs (no accumulation from a `while` loop), so this is also numerically
/// preferable to iterative `±2π` subtraction for large `|θ|`.
#[must_use]
#[inline]
pub fn wrap_to_pi(theta: f64) -> f64 {
    let r = theta.rem_euclid(TWO_PI);
    if r > std::f64::consts::PI {
        r - TWO_PI
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn wraps_into_principal_interval() {
        for k in -5..=5 {
            for &base in &[-3.0, -1.0, 0.0, 0.5, 2.9, PI - 0.01] {
                let theta = base + TWO_PI * k as f64;
                let w = wrap_to_pi(theta);
                assert!(
                    w > -PI - 1e-9 && w <= PI + 1e-9,
                    "wrap_to_pi({theta}) = {w} outside (-π, π]"
                );
                // Same equivalence class: differ by an integer multiple of 2π.
                let diff = (w - theta) / TWO_PI;
                assert!(
                    (diff - diff.round()).abs() < 1e-9,
                    "wrap changed the angle modulo 2π: {theta} -> {w}"
                );
            }
        }
    }

    #[test]
    fn fixed_points_and_boundaries() {
        assert!((wrap_to_pi(0.0)).abs() < 1e-12);
        assert!((wrap_to_pi(0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_to_pi(-0.5) + 0.5).abs() < 1e-12);
        // +π maps to +π (upper-closed); −π maps to +π (same class, principal rep).
        assert!((wrap_to_pi(PI) - PI).abs() < 1e-9);
        assert!((wrap_to_pi(-PI) - PI).abs() < 1e-9);
        // Anti-phase: a near-2π difference folds to a small negative angle.
        assert!((wrap_to_pi(TWO_PI - 0.1) + 0.1).abs() < 1e-9);
    }
}
