//! Coalescence operators (own construction — see the
//! [`population`](super) module docs for status).
//!
//! Two bubbles that merge pool their gas: the merged bubble holds
//! `n₁ + n₂` molecules and sits at the equilibrium radius for that
//! gas content (volume is conserved to the accuracy of the
//! equilibrium map's Laplace term). The probability that a bubble
//! has a neighbour close enough to merge during an expansion to
//! `R_max` is the geometric estimate of the formation model
//! (Build 5): neighbours are Poisson-distributed, and contact
//! requires a centre within `2·R_max`.
//!
//! **Physical-use warning (Build 7):** applying [`coalesce_group`]
//! every cycle with a freshly drawn pairing treats the population
//! as if bubbles re-randomise positions each cycle. For bubbles
//! fixed in tissue that is unphysical and it materially changes
//! outcomes (it was the only variant that produced a >10 µm tail in
//! the formation model — an artefact, not a result). Apply
//! one-shot, or track which bubbles are neighbours. During a burst
//! above MI ≈ 1, do not apply it at all: Duryea et al. (2015)
//! measured aggregation failing there (see
//! [`bjerknes_forces::regime`](crate::acoustics::bubble_dynamics::bjerknes_forces::regime)).

use super::{BubbleGroup, PopulationMedium};
use std::f64::consts::PI;

/// Probability that a bubble expanding to `r_max` [m] has at least
/// one neighbour in contact range, for `count` bubbles in
/// `focal_volume` [m³]: `1 − exp(−(4π/3)·(2R_max)³·N/V)`.
#[must_use]
pub fn neighbour_merge_probability(r_max: f64, count: f64, focal_volume: f64) -> f64 {
    if r_max <= 0.0 || count <= 0.0 || focal_volume <= 0.0 {
        return 0.0;
    }
    let lambda = (4.0 * PI / 3.0) * (2.0 * r_max).powi(3) * count / focal_volume;
    1.0 - (-lambda).exp()
}

/// Merge two single bubbles into one group of count 1: gas pools,
/// radius follows the equilibrium map.
#[must_use]
pub fn merge_pair(a: &BubbleGroup, b: &BubbleGroup, medium: &PopulationMedium) -> BubbleGroup {
    let gas = a.gas_per_bubble + b.gas_per_bubble;
    BubbleGroup {
        radius: medium.radius_at_gas(gas),
        count: 1.0,
        gas_per_bubble: gas,
    }
}

/// Merge a fraction of a monodisperse group with itself: `fraction`
/// of the bubbles pair up (two bubbles of gas `n` → one of gas
/// `2n`), the rest are unchanged. Gas and total bubble volume are
/// conserved (volume to the equilibrium map's Laplace accuracy).
/// One-shot operator — see the module warning.
#[must_use]
pub fn coalesce_group(
    group: &BubbleGroup,
    fraction: f64,
    medium: &PopulationMedium,
) -> Vec<BubbleGroup> {
    let f = fraction.clamp(0.0, 1.0);
    if f <= 0.0 {
        return vec![*group];
    }
    let merged_pairs = group.count * f / 2.0;
    let rest_count = group.count * (1.0 - f);
    let merged_gas = 2.0 * group.gas_per_bubble;
    let mut out = Vec::new();
    if rest_count > 0.0 {
        out.push(BubbleGroup {
            count: rest_count,
            ..*group
        });
    }
    if merged_pairs > 0.0 {
        out.push(BubbleGroup {
            radius: medium.radius_at_gas(merged_gas),
            count: merged_pairs,
            gas_per_bubble: merged_gas,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::bubble_dynamics::population::PopulationMedium;

    fn medium() -> PopulationMedium {
        PopulationMedium {
            ambient_pressure: 101_325.0,
            surface_tension: 0.056,
            temperature: 310.0,
        }
    }

    #[test]
    fn test_neighbour_probability_limits() {
        assert_eq!(neighbour_merge_probability(10e-6, 0.0, 6.28e-9), 0.0);
        // Sparse cloud: moderate probability. Formation values:
        // N=1054 per point in 6.28 mm³, R_max ≈ 51 µm →
        // λ = (4π/3)(2R)³N/V ≈ 0.75, p ≈ 0.53.
        let p = neighbour_merge_probability(51.2e-6, 1054.0, 6.28e-9);
        assert!(p > 0.45 && p < 0.60, "p = {p}");
        // Dense cloud saturates at 1.
        let p_dense = neighbour_merge_probability(100e-6, 1.0e6, 6.28e-9);
        assert!(p_dense > 0.99);
    }

    #[test]
    fn test_merge_pair_conserves_gas() {
        let m = medium();
        let a = BubbleGroup {
            radius: 3e-6,
            count: 1.0,
            gas_per_bubble: m.gas_at_radius(3e-6),
        };
        let merged = merge_pair(&a, &a, &m);
        assert!((merged.gas_per_bubble / (2.0 * a.gas_per_bubble) - 1.0).abs() < 1e-12);
        // Merged radius is larger, below the equal-volume 2^(1/3)
        // bound only through the Laplace term — check it is in a
        // sane band around 2^(1/3)·R.
        let expect = 3e-6 * 2.0_f64.powf(1.0 / 3.0);
        assert!(merged.radius > 0.9 * expect && merged.radius < 1.1 * expect);
    }

    #[test]
    fn test_coalesce_group_conserves_gas_and_reduces_count() {
        let m = medium();
        let g = BubbleGroup {
            radius: 5e-6,
            count: 1000.0,
            gas_per_bubble: m.gas_at_radius(5e-6),
        };
        let out = coalesce_group(&g, 0.5, &m);
        let gas: f64 = out.iter().map(BubbleGroup::total_gas).sum();
        assert!((gas / g.total_gas() - 1.0).abs() < 1e-9);
        let count: f64 = out.iter().map(|b| b.count).sum();
        assert!((count - 750.0).abs() < 1e-9);
    }
}
