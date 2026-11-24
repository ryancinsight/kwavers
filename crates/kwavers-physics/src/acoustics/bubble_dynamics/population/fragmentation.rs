//! Fragmentation operators (own construction — see the
//! [`population`](super) module docs for status).
//!
//! A fragmentation event replaces a parent bubble by several
//! daughters. The literature supplies only the *indicator* for when
//! fragmentation is expected — Chomas et al. (2001, *IEEE Trans.
//! UFFC* 48, 232–236) found fragmentation marked by a max/min
//! **diameter** ratio above 10, with >55% of bubbles fragmenting at
//! 2.4 MPa peak negative pressure versus <10% at 0.8 MPa — and no
//! daughter distribution. The kernels here are therefore explicit
//! modelling choices:
//!
//! **Single-sequence use only (Build 11, Nov 21, 2026).** These
//! kernels are calibrated/constructed for one exposure sequence
//! (one trigger + one burst). Iterating them per sequence over a
//! repeated exposure (the Iwasaki protocol repeats 200×) has no
//! fixed point: daughters stay above the collapse-ratio gate,
//! counts compound (×885 by sequence 10 in the prototype), and the
//! focal gas reservoir is exhausted by ~sequence 25–30, ending in
//! sub-µm dust. The kernel does not transfer across burst lengths
//! (Build 7) or across sequences (Build 11). A multi-sequence model
//! needs a removal/physics balance that does not currently exist —
//! the Build 11 removal inversion closed negative (holding the
//! fixed point at the measured ~3 µm residuals would need ≈24×
//! Epstein–Plesset dissolution; all candidate literature channels
//! together reach ~13% of the required removal) — so no
//! per-sequence removal term is provided here either.
//!
//! - [`LognormalKernel`]: daughter radii lognormal about a median
//!   ratio of the parent radius, truncated to ≤ the parent radius
//!   (a daughter cannot exceed its parent — the Build 5–7
//!   construction; this truncation is also why fragmentation alone
//!   can never grow a large-radius tail).
//! - [`EqualSplitKernel`]: each fragmenting parent splits into a
//!   fixed number of equal-volume daughters.
//!
//! Both conserve gas exactly: daughters' equilibrium gas contents
//! (through [`PopulationMedium`]) sum to the parent's gas, with the
//! daughter count following from the gas assigned to each size.

use super::{BubbleGroup, PopulationMedium};

/// Chomas et al. (2001) fragmentation indicator: max/min diameter
/// ratio above which fragmentation is expected. Diameter and radius
/// ratios are identical, so the ratio is computed on radii.
pub const CHOMAS_FRAGMENTATION_DIAMETER_RATIO: f64 = 10.0;

/// Per-cycle fragmentation probability **FITTED** to Li et al.
/// (2021) in the prototype's Stage 7 kernel-B construction. This is
/// a calibration, not a literature value and not a prediction; it is
/// exported only so the fitted value has one documented home.
pub const P_FRAG_FITTED_LI_2021: f64 = 0.02;

/// Collapse ratio `X = R_max / R_min` of a bubble cycle. Compare
/// against [`CHOMAS_FRAGMENTATION_DIAMETER_RATIO`] for the
/// fragmentation indicator. `r_min` should be the pressure-balance
/// radius at peak compression, not the equilibrium radius.
#[must_use]
pub fn collapse_ratio(r_max: f64, r_min: f64) -> f64 {
    if r_min <= 0.0 {
        return f64::INFINITY;
    }
    r_max / r_min
}

/// A daughter-distribution kernel: turns the fragmenting fraction
/// of a group into daughter groups. Implementations must conserve
/// total gas exactly (to rounding).
pub trait FragmentationKernel {
    /// Fragment `fraction` (0–1) of `group`'s bubbles; the
    /// unfragmented remainder is included in the output.
    fn fragment(
        &self,
        group: &BubbleGroup,
        fraction: f64,
        medium: &PopulationMedium,
    ) -> Vec<BubbleGroup>;
}

/// Equal-volume split: each fragmenting parent becomes
/// `daughters` bubbles of radius `R·k^(−1/3)`, each holding `n/k`
/// molecules. (The daughter radius implied by equal volume and the
/// equilibrium radius implied by `n/k` differ slightly through the
/// Laplace term; the operator keeps the gas split exact and sets
/// the radius from the equilibrium map for `n/k`, so radius and gas
/// stay consistent.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqualSplitKernel {
    /// Daughters per fragmenting parent (≥ 1).
    pub daughters: f64,
}

impl FragmentationKernel for EqualSplitKernel {
    fn fragment(
        &self,
        group: &BubbleGroup,
        fraction: f64,
        medium: &PopulationMedium,
    ) -> Vec<BubbleGroup> {
        let f = fraction.clamp(0.0, 1.0);
        if f <= 0.0 || self.daughters <= 1.0 {
            return vec![*group];
        }
        let frag_count = group.count * f;
        let rest_count = group.count * (1.0 - f);
        let daughter_gas = group.gas_per_bubble / self.daughters;
        let mut out = Vec::new();
        if rest_count > 0.0 {
            out.push(BubbleGroup {
                count: rest_count,
                ..*group
            });
        }
        out.push(BubbleGroup {
            radius: medium.radius_at_gas(daughter_gas),
            count: frag_count * self.daughters,
            gas_per_bubble: daughter_gas,
        });
        out
    }
}

/// Lognormal daughter kernel: daughter radii are lognormal with
/// median `median_ratio` × parent radius and log-width `sigma`,
/// **truncated to ≤ the parent radius**, discretised onto
/// `n_bins` log-spaced bins between `median_ratio·exp(−3σ)` and 1
/// (in units of the parent radius). Each fragmenting parent's gas
/// is distributed over the bins in proportion to the kernel pdf;
/// a bin's bubble count is its gas divided by the equilibrium gas
/// at the bin radius, so gas is conserved exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LognormalKernel {
    /// Median daughter radius as a fraction of the parent radius.
    pub median_ratio: f64,
    /// Log-width (σ of ln radius).
    pub sigma: f64,
    /// Number of discretisation bins.
    pub n_bins: usize,
}

impl LognormalKernel {
    /// Kernel weights and bin radius ratios (relative to the
    /// parent radius): bin centres log-spaced over the truncated
    /// support, weights from the lognormal pdf, normalised.
    fn bins(&self) -> Vec<(f64, f64)> {
        let lo = (self.median_ratio * (-3.0 * self.sigma).exp()).min(0.999);
        let hi = 1.0_f64;
        let n = self.n_bins.max(1);
        let mut out = Vec::with_capacity(n);
        let mut total = 0.0;
        for i in 0..n {
            let frac = (i as f64 + 0.5) / n as f64;
            let ratio = (lo.ln() + frac * (hi.ln() - lo.ln())).exp();
            let z = (ratio.ln() - self.median_ratio.ln()) / self.sigma;
            // Lognormal pdf in ratio space (the 1/ratio Jacobian is
            // part of the density over radius).
            let w = (-0.5 * z * z).exp() / ratio;
            out.push((ratio, w));
            total += w;
        }
        if total > 0.0 {
            for (_, w) in &mut out {
                *w /= total;
            }
        }
        out
    }
}

impl FragmentationKernel for LognormalKernel {
    fn fragment(
        &self,
        group: &BubbleGroup,
        fraction: f64,
        medium: &PopulationMedium,
    ) -> Vec<BubbleGroup> {
        let f = fraction.clamp(0.0, 1.0);
        if f <= 0.0 {
            return vec![*group];
        }
        let frag_count = group.count * f;
        let rest_count = group.count * (1.0 - f);
        let frag_gas_total = frag_count * group.gas_per_bubble;
        let mut out = Vec::new();
        if rest_count > 0.0 {
            out.push(BubbleGroup {
                count: rest_count,
                ..*group
            });
        }
        for (ratio, w) in self.bins() {
            if w <= 0.0 {
                continue;
            }
            let radius = ratio * group.radius;
            let gas_here = frag_gas_total * w;
            let per = medium.gas_at_radius(radius);
            if per <= 0.0 {
                continue;
            }
            out.push(BubbleGroup {
                radius,
                count: gas_here / per,
                gas_per_bubble: per,
            });
        }
        out
    }
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

    fn group() -> BubbleGroup {
        let m = medium();
        BubbleGroup {
            radius: 5e-6,
            count: 1000.0,
            gas_per_bubble: m.gas_at_radius(5e-6),
        }
    }

    #[test]
    fn test_collapse_ratio_and_indicator() {
        // Formation-regime values (Build 7): X ≈ 37 (Kuji) / 23
        // (Iwasaki) — above the Chomas indicator.
        assert!(collapse_ratio(51.2e-6, 1.4e-6) > CHOMAS_FRAGMENTATION_DIAMETER_RATIO);
        assert!((collapse_ratio(10e-6, 2e-6) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_equal_split_conserves_gas() {
        let g = group();
        let out = EqualSplitKernel { daughters: 8.0 }.fragment(&g, 1.0, &medium());
        let gas: f64 = out.iter().map(BubbleGroup::total_gas).sum();
        assert!((gas / g.total_gas() - 1.0).abs() < 1e-9);
        let count: f64 = out.iter().map(|b| b.count).sum();
        assert!((count - 8000.0).abs() < 1e-6);
        // Daughters are smaller than the parent.
        assert!(out.iter().all(|b| b.radius < g.radius));
    }

    #[test]
    fn test_lognormal_conserves_gas_and_truncates() {
        let g = group();
        let kernel = LognormalKernel {
            median_ratio: 0.6,
            sigma: 0.3,
            n_bins: 24,
        };
        let out = kernel.fragment(&g, 0.75, &medium());
        let gas: f64 = out.iter().map(BubbleGroup::total_gas).sum();
        assert!((gas / g.total_gas() - 1.0).abs() < 1e-9);
        // No daughter exceeds the parent radius (truncation).
        assert!(out.iter().all(|b| b.radius <= g.radius * (1.0 + 1e-12)));
        // Fragmentation increases the bubble count.
        let count: f64 = out.iter().map(|b| b.count).sum();
        assert!(count > g.count);
    }

    #[test]
    fn test_zero_fraction_is_identity() {
        let g = group();
        let out = EqualSplitKernel { daughters: 4.0 }.fragment(&g, 0.0, &medium());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], g);
    }
}
