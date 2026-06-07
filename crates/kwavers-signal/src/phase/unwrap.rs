//! Phase unwrapping.
//!
//! Recovers a continuous phase field from one wrapped into the principal range
//! `(-π, π]`. Used by MR/phase-gradient elastography, Doppler phase tracking, and
//! interferometric reconstruction.
//!
//! The 2-D routine is the separable (Itoh) path-following unwrapper: unwrap the
//! first column, then unwrap each row seeded from that column. It is **exact** for
//! residue-free fields (smooth wavefields sampled below the Nyquist phase rate,
//! i.e. adjacent-sample phase steps `< π`). For fields with residues (noise,
//! aliasing) the result is path-dependent; use the residue-aware tools in
//! [`super::goldstein`] (residue detection + masked flood-fill unwrap) for those.
//!
//! # References
//! - Itoh, K. (1982). "Analysis of the phase unwrapping algorithm."
//!   *Applied Optics*, 21(14), 2470.
//! - Ghiglia, D. C., & Pritt, M. D. (1998). *Two-Dimensional Phase Unwrapping*.

use core::f64::consts::TAU;
use ndarray::Array2;

/// Wrap a phase difference into the principal interval `(-π, π]`.
#[inline]
fn wrap_to_pi(d: f64) -> f64 {
    d - TAU * (d / TAU).round()
}

/// Unwrap a 1-D wrapped phase sequence (Itoh).
///
/// The first sample is preserved; each subsequent sample adds the principal-range
/// wrapped first difference. Reconstructs the continuous phase up to the (already
/// correct) starting value, provided successive true-phase steps are `< π` in
/// magnitude.
#[must_use]
pub fn unwrap_1d(wrapped: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(wrapped.len());
    let Some(&first) = wrapped.first() else {
        return out;
    };
    out.push(first);
    for i in 1..wrapped.len() {
        let step = wrap_to_pi(wrapped[i] - wrapped[i - 1]);
        out.push(out[i - 1] + step);
    }
    out
}

/// Unwrap a 2-D wrapped phase field by the separable (Itoh) path:
/// unwrap column 0 down the rows, then unwrap each row seeded from that column.
///
/// Exact for residue-free fields; `out[[0, 0]] == wrapped[[0, 0]]` (the absolute
/// phase offset is preserved at the anchor sample).
#[must_use]
pub fn unwrap_2d(wrapped: &Array2<f64>) -> Array2<f64> {
    let (nr, nc) = wrapped.dim();
    let mut out = Array2::zeros((nr, nc));
    if nr == 0 || nc == 0 {
        return out;
    }

    // 1. Unwrap the first column along the row direction.
    let col0: Vec<f64> = (0..nr).map(|r| wrapped[[r, 0]]).collect();
    let col0_u = unwrap_1d(&col0);
    for (r, &v) in col0_u.iter().enumerate() {
        out[[r, 0]] = v;
    }

    // 2. Unwrap each row across columns, re-anchored to the unwrapped column 0.
    for r in 0..nr {
        let row: Vec<f64> = (0..nc).map(|c| wrapped[[r, c]]).collect();
        let row_u = unwrap_1d(&row);
        let offset = out[[r, 0]] - row_u[0];
        for c in 0..nc {
            out[[r, c]] = row_u[c] + offset;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn wrap(x: f64) -> f64 {
        x - TAU * (x / TAU).round()
    }

    #[test]
    fn unwrap_1d_recovers_linear_ramp() {
        // true phase = 0.7·i (step 0.7 < π), wrapped into (-π, π]
        let truth: Vec<f64> = (0..40).map(|i| 0.7 * i as f64).collect();
        let wrapped: Vec<f64> = truth.iter().map(|&x| wrap(x)).collect();
        let unwrapped = unwrap_1d(&wrapped);
        // anchored at wrap(0) = 0, so recovery is exact (not just up to a constant)
        for (u, t) in unwrapped.iter().zip(&truth) {
            assert!((u - t).abs() < 1e-9, "unwrap {u} != truth {t}");
        }
        // the wrapped signal genuinely had ≥1 jump (otherwise the test is trivial)
        assert!(wrapped
            .windows(2)
            .any(|w| (w[1] - w[0]).abs() > std::f64::consts::PI));
    }

    #[test]
    fn unwrap_2d_recovers_plane() {
        // true phase = 0.4·r + 0.55·c (both steps < π), φ(0,0) = 0
        let (nr, nc) = (12, 16);
        let mut wrapped = Array2::zeros((nr, nc));
        let mut truth = Array2::zeros((nr, nc));
        for r in 0..nr {
            for c in 0..nc {
                let phi = 0.4 * r as f64 + 0.55 * c as f64;
                truth[[r, c]] = phi;
                wrapped[[r, c]] = wrap(phi);
            }
        }
        let out = unwrap_2d(&wrapped);
        let max_err = out
            .iter()
            .zip(truth.iter())
            .map(|(o, t)| (o - t).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-9, "2-D unwrap max error {max_err} too large");
    }

    #[test]
    fn unwrap_is_identity_on_already_continuous_phase() {
        // smooth phase already within range and residue-free → unchanged
        let smooth: Vec<f64> = (0..10).map(|i| 0.1 * i as f64).collect();
        let out = unwrap_1d(&smooth);
        for (o, s) in out.iter().zip(&smooth) {
            assert!((o - s).abs() < 1e-12);
        }
    }

    #[test]
    fn empty_inputs_are_handled() {
        assert!(unwrap_1d(&[]).is_empty());
        assert_eq!(unwrap_2d(&Array2::<f64>::zeros((0, 0))).dim(), (0, 0));
    }
}
