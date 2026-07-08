//! Residue-aware 2-D phase unwrapping.
//!
//! The separable Itoh unwrapper ([`super::unwrap_2d`]) is exact only for
//! *residue-free* fields; where the wrapped phase contains **residues** (`±2π`
//! circulation around a 2×2 loop, from noise/aliasing/vortices) the path-following
//! result becomes path-dependent. This module provides the residue-aware tools:
//!
//! - [`phase_residues`] — the Goldstein step-1 diagnostic: the residue charge of
//!   every 2×2 plaquette (`+1`, `0`, `−1`). A field with no residues is safe for
//!   Itoh unwrapping.
//! - [`residue_count`] / [`is_unwrap_reliable`] — quick gates.
//! - [`masked_unwrap_2d`] — flood-fill (BFS) path-following unwrap restricted to a
//!   validity mask, so the caller can exclude unreliable regions (e.g. a dilation
//!   around residues) and unwrap the trustworthy remainder consistently.
//!
//! - [`goldstein_branch_cut_mask`] — automatic branch-cut placement: greedily
//!   joins each residue to the nearest opposite-charge residue (leftovers to the
//!   border) with a Manhattan chain of excluded pixels, so no valid 4-connected
//!   loop encircles a net residue.
//! - [`goldstein_unwrap_2d`] — `masked_unwrap_2d` over that auto-generated mask: a
//!   full residue-aware unwrapper (residue-free fields reduce to the Itoh path).
//!
//! # References
//! - Goldstein, R. M., Zebker, H. A., Werner, C. L. (1988). "Satellite radar
//!   interferometry: Two-dimensional phase unwrapping." *Radio Science*, 23(4).
//! - Ghiglia, D. C., & Pritt, M. D. (1998). *Two-Dimensional Phase Unwrapping*.

use core::f64::consts::TAU;
use leto::Array2;
use std::collections::VecDeque;

#[inline]
fn wrap_to_pi(d: f64) -> f64 {
    d - TAU * (d / TAU).round()
}

/// Residue map of a wrapped phase field: for each 2×2 plaquette, the signed sum of
/// the four wrapped phase differences around the loop, in units of `2π`
/// (`+1`/`0`/`−1`). Output shape is `(nr−1, nc−1)`; a residue-free field is all 0.
#[must_use]
pub fn phase_residues(wrapped: &Array2<f64>) -> Array2<i32> {
    let [nr, nc] = wrapped.shape();
    if nr < 2 || nc < 2 {
        return Array2::zeros([nr.saturating_sub(1), nc.saturating_sub(1)]);    }
    let mut res = Array2::zeros([nr - 1, nc - 1]);
    for r in 0..nr - 1 {
        for c in 0..nc - 1 {
            // loop (r,c)→(r,c+1)→(r+1,c+1)→(r+1,c)→(r,c)
            let d1 = wrap_to_pi(wrapped[[r, c + 1]] - wrapped[[r, c]]);
            let d2 = wrap_to_pi(wrapped[[r + 1, c + 1]] - wrapped[[r, c + 1]]);
            let d3 = wrap_to_pi(wrapped[[r + 1, c]] - wrapped[[r + 1, c + 1]]);
            let d4 = wrap_to_pi(wrapped[[r, c]] - wrapped[[r + 1, c]]);
            let circulation = d1 + d2 + d3 + d4;
            res[[r, c]] = (circulation / TAU).round() as i32;
        }
    }
    res
}

/// Number of non-zero residues in a wrapped phase field.
#[must_use]
pub fn residue_count(wrapped: &Array2<f64>) -> usize {
    phase_residues(wrapped).iter().filter(|&&r| r != 0).count()
}

/// `true` when the field is residue-free and therefore safe for the path-following
/// [`super::unwrap_2d`].
#[must_use]
pub fn is_unwrap_reliable(wrapped: &Array2<f64>) -> bool {
    residue_count(wrapped) == 0
}

/// Flood-fill (BFS) path-following unwrap restricted to a validity `mask`
/// (`true` = unwrap this pixel). Unwrapping starts from the first valid pixel and
/// propagates only between adjacent valid pixels, so a mask that excludes
/// residue regions yields a consistent unwrap of the trustworthy area. Masked or
/// unreachable pixels are returned as `f64::NAN`.
#[must_use]
pub fn masked_unwrap_2d(wrapped: &Array2<f64>, mask: &Array2<bool>) -> Array2<f64> {
    let [nr, nc] = wrapped.shape();
    let mut out = Array2::from_elem([nr, nc], f64::NAN);
    if nr == 0 || nc == 0 || mask.shape() != [nr, nc] {
        return out;
    }
    // find the first valid seed
    let seed = (0..nr)
        .flat_map(|r| (0..nc).map(move |c| (r, c)))
        .find(|&(r, c)| mask[[r, c]]);
    let Some((sr, sc)) = seed else {
        return out;
    };
    out[[sr, sc]] = wrapped[[sr, sc]];
    let mut queue = VecDeque::new();
    queue.push_back((sr, sc));
    while let Some((r, c)) = queue.pop_front() {
        let cur = out[[r, c]];
        let mut neigh: Vec<(usize, usize)> = Vec::with_capacity(4);
        if r > 0 {
            neigh.push((r - 1, c));
        }
        if r + 1 < nr {
            neigh.push((r + 1, c));
        }
        if c > 0 {
            neigh.push((r, c - 1));
        }
        if c + 1 < nc {
            neigh.push((r, c + 1));
        }
        for (nr_, nc_) in neigh {
            if mask[[nr_, nc_]] && out[[nr_, nc_]].is_nan() {
                out[[nr_, nc_]] = cur + wrap_to_pi(wrapped[[nr_, nc_]] - cur);
                queue.push_back((nr_, nc_));
            }
        }
    }
    out
}

/// Automatic Goldstein branch-cut validity mask. Detects residues and **grounds
/// each to the nearest border** with a straight chain of *excluded* pixels, so no
/// valid 4-connected loop can encircle a residue (any such loop would have to cross
/// a grounded cut). Grounding every residue independently is robust to clustered
/// residues (unlike greedy pairing). Returns the mask (`true` = unwrap, `false` =
/// on a cut).
#[must_use]
pub fn goldstein_branch_cut_mask(wrapped: &Array2<f64>) -> Array2<bool> {
    let [nr, nc] = wrapped.shape();
    let mut mask = Array2::from_elem([nr, nc], true);
    if nr < 2 || nc < 2 {
        return mask;
    }
    let residues = phase_residues(wrapped);
    for ([pr, pc], &q) in residues.indexed_iter() {
        if q == 0 {
            continue;
        }
        // anchor pixel column/row for the cut (a corner of the residue plaquette)
        let (ar, ac) = (pr + 1, pc + 1);
        let (to_top, to_bot, to_left, to_right) = (ar, nr - 1 - ar, ac, nc - 1 - ac);
        let m = to_top.min(to_bot).min(to_left).min(to_right);
        if m == to_top {
            for k in 0..=ar {
                mask[[k, ac]] = false;
            }
        } else if m == to_bot {
            for k in ar..nr {
                mask[[k, ac]] = false;
            }
        } else if m == to_left {
            for k in 0..=ac {
                mask[[ar, k]] = false;
            }
        } else {
            for k in ac..nc {
                mask[[ar, k]] = false;
            }
        }
    }
    mask
}

/// Full residue-aware Goldstein unwrap: build the branch-cut mask and flood-fill.
/// Residue-free fields reduce to the Itoh path; cut pixels are returned as `NaN`.
#[must_use]
pub fn goldstein_unwrap_2d(wrapped: &Array2<f64>) -> Array2<f64> {
    let mask = goldstein_branch_cut_mask(wrapped);
    masked_unwrap_2d(wrapped, &mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn wrap(x: f64) -> f64 {
        x - TAU * (x / TAU).round()
    }

    #[test]
    fn smooth_field_has_no_residues() {
        let (nr, nc) = (16, 16);
        let mut w = Array2::zeros([nr, nc]);
        for r in 0..nr {
            for c in 0..nc {
                w[[r, c]] = wrap(0.3 * r as f64 + 0.45 * c as f64);
            }
        }
        assert_eq!(residue_count(&w), 0);
        assert!(is_unwrap_reliable(&w));
    }

    #[test]
    fn phase_vortex_has_a_unit_residue() {
        // a discrete phase singularity φ = atan2(r−rc, c−cc) carries one ±1 residue
        let (nr, nc) = (16, 16);
        let (rc, cc) = (7.5, 7.5);
        let mut w = Array2::zeros([nr, nc]);
        for r in 0..nr {
            for c in 0..nc {
                w[[r, c]] = (r as f64 - rc).atan2(c as f64 - cc); // already in (−π, π]
            }
        }
        let res = phase_residues(&w);
        let total: i32 = res.iter().sum();
        let nonzero = res.iter().filter(|&&v| v != 0).count();
        assert_eq!(
            nonzero, 1,
            "a single vortex → exactly one residue plaquette"
        );
        assert_eq!(total.abs(), 1, "residue charge ±1");
        assert!(!is_unwrap_reliable(&w));
    }

    #[test]
    fn masked_unwrap_recovers_plane_and_rewrap_is_consistent() {
        let (nr, nc) = (12, 14);
        let mut w = Array2::zeros([nr, nc]);
        let mut truth = Array2::zeros([nr, nc]);
        for r in 0..nr {
            for c in 0..nc {
                let phi = 0.4 * r as f64 + 0.5 * c as f64;
                truth[[r, c]] = phi;
                w[[r, c]] = wrap(phi);
            }
        }
        // exclude a small block (as if masking a residue region)
        let mut mask = Array2::from_elem([nr, nc], true);
        mask[[5, 6]] = false;
        mask[[5, 7]] = false;

        let out = masked_unwrap_2d(&w, &mask);
        for r in 0..nr {
            for c in 0..nc {
                if mask[[r, c]] {
                    // exact recovery on the valid region (φ(0,0)=0 anchor)
                    assert!(
                        (out[[r, c]] - truth[[r, c]]).abs() < 1e-9,
                        "recover ({r},{c})"
                    );
                    // re-wrap consistency: only multiples of 2π were added
                    assert!((wrap(out[[r, c]]) - w[[r, c]]).abs() < 1e-9);
                } else {
                    assert!(out[[r, c]].is_nan(), "masked pixel must be NaN");
                }
            }
        }
    }

    #[test]
    fn goldstein_unwrap_residue_free_recovers_plane() {
        let (nr, nc) = (12, 14);
        let mut w = Array2::zeros([nr, nc]);
        let mut truth = Array2::zeros([nr, nc]);
        for r in 0..nr {
            for c in 0..nc {
                let phi = 0.4 * r as f64 + 0.5 * c as f64;
                truth[[r, c]] = phi;
                w[[r, c]] = wrap(phi);
            }
        }
        // no residues → mask all-valid → reduces to the Itoh path
        assert!(goldstein_branch_cut_mask(&w).iter().all(|&v| v));
        let out = goldstein_unwrap_2d(&w);
        for r in 0..nr {
            for c in 0..nc {
                assert!((out[[r, c]] - truth[[r, c]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn goldstein_unwrap_handles_a_phase_dipole_consistently() {
        // two opposite vortices → a +1 and a −1 residue
        let (nr, nc) = (24, 24);
        let (r1, c1) = (8.0, 8.0);
        let (r2, c2) = (16.0, 16.0);
        let mut w = Array2::zeros([nr, nc]);
        for r in 0..nr {
            for c in 0..nc {
                let p1 = (r as f64 - r1).atan2(c as f64 - c1);
                let p2 = (r as f64 - r2).atan2(c as f64 - c2);
                w[[r, c]] = wrap(p1 - p2);
            }
        }
        assert!(residue_count(&w) >= 2, "dipole should have residues");
        let mask = goldstein_branch_cut_mask(&w);
        assert!(mask.iter().any(|&v| !v), "a branch cut must be placed");

        let out = goldstein_unwrap_2d(&w);
        // Branch-cut correctness = continuity: with residues grounded by cuts, no
        // valid loop encircles a residue, so the unwrapped field has NO 2π seam —
        // every adjacent valid–valid pair differs by < π. (Without cuts a 2π jump
        // line would emanate from the residue.)
        let mut unwrapped = 0;
        for r in 0..nr {
            for c in 0..nc {
                if out[[r, c]].is_nan() {
                    continue;
                }
                unwrapped += 1;
                if r + 1 < nr && !out[[r + 1, c]].is_nan() {
                    assert!(
                        (out[[r + 1, c]] - out[[r, c]]).abs() < PI,
                        "row seam at ({r},{c})"
                    );
                }
                if c + 1 < nc && !out[[r, c + 1]].is_nan() {
                    assert!(
                        (out[[r, c + 1]] - out[[r, c]]).abs() < PI,
                        "col seam at ({r},{c})"
                    );
                }
            }
        }
        // most of the field is unwrapped (only the thin grounded cuts are excluded)
        assert!(
            unwrapped as f64 > 0.85 * (nr * nc) as f64,
            "only {unwrapped} unwrapped"
        );
    }

    #[test]
    fn degenerate_inputs() {
        assert_eq!(phase_residues(&Array2::<f64>::zeros([1, 5])).shape(), [0, 4]);
        let w = Array2::<f64>::zeros([3, 3]);
        let m = Array2::from_elem([3, 3], false);
        assert!(masked_unwrap_2d(&w, &m).iter().all(|v| v.is_nan()));
    }
}
