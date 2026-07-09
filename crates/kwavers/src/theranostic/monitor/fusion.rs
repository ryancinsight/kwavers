//! Multi-modal lesion fusion.
//!
//! The monitor produces two independent images of the same slice from the same
//! recording window: the FD-CBS **quantitative** sound-speed-change map
//! ([`super::fd`]) and the PAM **passive** cavitation source map
//! ([`super::pam`]). Each has different failure modes — FWI is quantitative but
//! ill-posed at the boundary; PAM is robust and through-skull but only energy,
//! not Δc. Fusing them yields a lesion image that is trusted where the modalities
//! *agree*, which is the decision surface for "is there a lesion here, and how
//! confident am I".
//!
//! Two fused products (both over the slice grid, normalized to `[0, 1]`):
//! - **agreement** = `√(q̂ · p̂)` — geometric mean; high only where *both* the
//!   quantitative and passive maps are strong (a confirmed lesion), zero if either
//!   is absent. This is the conservative, low-false-positive channel.
//! - **union** = `max(q̂, p̂)` — either modality firing; the sensitive channel.
//!
//! `q̂`, `p̂` are each min-max normalized (`|Δc|` for the signed quantitative map,
//! raw energy for PAM).

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2;

use crate::parallel::zip_two_mut_two_refs;

/// Relative trust placed in each modality when forming the union channel.
#[derive(Clone, Copy, Debug)]
pub struct FusionWeights {
    /// Weight on the FD-CBS quantitative `|Δc|` map.
    pub quantitative: f64,
    /// Weight on the PAM passive source map.
    pub passive: f64,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            quantitative: 1.0,
            passive: 1.0,
        }
    }
}

/// Fused lesion products over the monitored slice.
#[derive(Clone, Debug)]
pub struct FusedLesion {
    /// Agreement (geometric-mean) confidence, `[0, 1]` — confirmed by both.
    pub agreement: Array2<f64>,
    /// Union (weighted-max) confidence, `[0, 1]` — either modality.
    pub union: Array2<f64>,
}

/// Min-max normalize to `[0, 1]`; an all-equal map returns all zeros (no signal).
fn normalize_abs(map: &Array2<f64>) -> Array2<f64> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in map.iter() {
        let a = v.abs();
        lo = lo.min(a);
        hi = hi.max(a);
    }
    let span = hi - lo;
    // `span > 0.0` is false for a zero/negative span and for NaN, so the else
    // branch (all-zeros, "no signal") covers the degenerate cases.
    if span > 0.0 {
        map.mapv(|v| (v.abs() - lo) / span)
    } else {
        Array2::zeros(map.raw_dim())
    }
}

/// Fuse the quantitative Δc map and the passive PAM map into agreement and union
/// lesion-confidence images.
///
/// `quantitative_dc` and `passive_energy` must share the slice shape.
///
/// # Errors
/// Returns [`KwaversError::DimensionMismatch`] if the two maps differ in shape.
pub fn fuse_lesion_map(
    quantitative_dc: &Array2<f64>,
    passive_energy: &Array2<f64>,
    weights: FusionWeights,
) -> KwaversResult<FusedLesion> {
    if quantitative_dc.dim() != passive_energy.dim() {
        return Err(KwaversError::DimensionMismatch(format!(
            "fusion maps differ: quantitative {:?} vs passive {:?}",
            quantitative_dc.dim(),
            passive_energy.dim()
        )));
    }
    let q = normalize_abs(quantitative_dc);
    let p = normalize_abs(passive_energy);

    let wq = weights.quantitative.max(0.0);
    let wp = weights.passive.max(0.0);
    // Weighted max normalized by the larger weight keeps the union in [0, 1]
    // (each normalized map is already in [0, 1]).
    let w_max = wq.max(wp).max(f64::EPSILON);

    let mut agreement = Array2::zeros(q.raw_dim());
    let mut union = Array2::zeros(q.raw_dim());
    zip_two_mut_two_refs(
        agreement.view_mut(),
        union.view_mut(),
        q.view(),
        p.view(),
        |a, u, &qv, &pv| {
            *a = (qv * pv).sqrt();
            *u = (wq * qv).max(wp * pv) / w_max;
        },
    );
    Ok(FusedLesion { agreement, union })
}

/// Count pixels whose confidence is at or above `threshold` — a lesion-extent
/// proxy for tracking growth across monitor frames.
#[must_use]
pub fn lesion_extent(confidence: &Array2<f64>, threshold: f64) -> usize {
    confidence.iter().filter(|&&v| v >= threshold).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array2;

    fn peak_at(n: usize, i: usize, j: usize, val: f64) -> Array2<f64> {
        let mut m = Array2::zeros((n, n));
        m[[i, j]] = val;
        m
    }

    #[test]
    fn agreement_high_only_where_both_fire() {
        let n = 8;
        let q = peak_at(n, 3, 4, 50.0); // quantitative lesion at (3,4)
        let p = peak_at(n, 3, 4, 1e6); // passive source at (3,4)
        let fused = fuse_lesion_map(&q, &p, FusionWeights::default()).unwrap();
        assert!(
            (fused.agreement[[3, 4]] - 1.0).abs() < 1e-9,
            "co-located → agreement 1"
        );
        // Everywhere else both are zero → agreement zero.
        assert!(fused.agreement.iter().filter(|&&v| v > 1e-9).count() == 1);
    }

    #[test]
    fn disagreement_suppresses_agreement_but_union_keeps_both() {
        let n = 8;
        let q = peak_at(n, 1, 1, 50.0); // quantitative at (1,1)
        let p = peak_at(n, 6, 6, 1e6); // passive at (6,6) — different pixel
        let fused = fuse_lesion_map(&q, &p, FusionWeights::default()).unwrap();
        // No pixel has both → agreement is ~0 everywhere.
        assert!(
            fused.agreement.iter().all(|&v| v < 1e-9),
            "no co-location → no agreement"
        );
        // Union shows both peaks.
        assert!(fused.union[[1, 1]] > 0.5);
        assert!(fused.union[[6, 6]] > 0.5);
    }

    #[test]
    fn signed_quantitative_uses_magnitude() {
        // A negative Δc (cavitation Wood collapse) must register as a lesion.
        let n = 6;
        let q = peak_at(n, 2, 2, -400.0);
        let p = peak_at(n, 2, 2, 5.0);
        let fused = fuse_lesion_map(&q, &p, FusionWeights::default()).unwrap();
        assert!(
            (fused.agreement[[2, 2]] - 1.0).abs() < 1e-9,
            "negative Δc is still a lesion"
        );
    }

    #[test]
    fn zero_maps_yield_zero_confidence() {
        let n = 5;
        let q = Array2::<f64>::zeros((n, n));
        let p = Array2::<f64>::zeros((n, n));
        let fused = fuse_lesion_map(&q, &p, FusionWeights::default()).unwrap();
        assert!(fused.agreement.iter().all(|&v| v == 0.0));
        assert!(fused.union.iter().all(|&v| v == 0.0));
        assert_eq!(lesion_extent(&fused.agreement, 0.5), 0);
    }

    #[test]
    fn lesion_extent_counts_growth() {
        let mut c = Array2::<f64>::zeros((4, 4));
        c[[0, 0]] = 0.9;
        c[[0, 1]] = 0.6;
        c[[1, 1]] = 0.4;
        assert_eq!(lesion_extent(&c, 0.5), 2);
        assert_eq!(lesion_extent(&c, 0.3), 3);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let q = Array2::<f64>::zeros((4, 4));
        let p = Array2::<f64>::zeros((4, 5));
        assert!(fuse_lesion_map(&q, &p, FusionWeights::default()).is_err());
    }

    /// End-to-end hybrid monitor: FD-CBS quantitative + PAM passive → fusion,
    /// all on the same synthetic lesion slice, must localize the lesion.
    #[test]
    fn hybrid_pipeline_localizes_lesion_from_both_channels() {
        use super::super::{fd, pam};
        use leto::Array3;

        let n = 12usize;
        let centre = n / 2; // 6
        let spacing = 1.0e-3;

        // ── Quantitative channel: FD-CBS reconstruct a +60 m/s lesion. ──
        let cfg = fd::FdMonitorConfig {
            ring_elements: 16,
            ring_diameter_m: 0.018,
            spacing_m: spacing,
            frequencies_hz: vec![3.0e5, 5.0e5],
            reference_sound_speed_m_s: 1500.0,
            min_sound_speed_m_s: 1400.0,
            max_sound_speed_m_s: 1700.0,
            fwi_iterations: 6,
            estimate_source_scaling: false,
            cbs_iterations: 20,
            cbs_tolerance: 1.0e-3,
            tikhonov_weight: 0.0,
            use_gauss_newton: true,
        };
        let mut background = Array3::from_elem((n, n, 1), 1500.0);
        let mut perturbed = background.clone();
        for i in centre - 1..=centre + 1 {
            for j in centre - 1..=centre + 1 {
                perturbed[[i, j, 0]] = 1560.0;
            }
        }
        let ring = fd::ring_around_slice(cfg.ring_elements, cfg.ring_diameter_m).unwrap();
        let recon = fd::reconstruct(&perturbed, &background, &ring, &cfg).unwrap();
        let mut quant = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                quant[[i, j]] = recon[[i, j, 0]] - 1500.0;
            }
        }
        background.fill(0.0); // silence unused-mut lint without affecting recon

        // ── Passive channel: PAM map of the cavitation emission at the lesion. ──
        // Slice physical frame matches the FD voxel-centre convention.
        let origin = [
            (0.5 - n as f64 / 2.0) * spacing,
            0.0,
            (0.5 - n as f64 / 2.0) * spacing,
        ];
        let pam_cfg = pam::PamMonitorConfig {
            sound_speed_m_s: 1500.0,
            origin_m: origin,
            spacing_m: spacing,
            nx: n,
            nz: n,
        };
        let r = 0.009;
        let elements: Vec<[f64; 3]> = (0..16)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 16.0;
                [r * a.cos(), 0.0, r * a.sin()]
            })
            .collect();
        let source = [
            origin[0] + centre as f64 * spacing,
            0.0,
            origin[2] + centre as f64 * spacing,
        ];
        let fs = 2.0e6;
        let data = pam::synthesize_emission(source, &elements, fs, 1500.0, 400, 3.0).unwrap();
        let passive = pam::passive_acoustic_map(&data, &elements, fs, &pam_cfg).unwrap();

        // ── Fusion. ──
        let fused = fuse_lesion_map(&quant, &passive, FusionWeights::default()).unwrap();

        // Agreement peak must localize to the lesion region (indices 5..=7).
        let (mut pi, mut pj, mut pv) = (0usize, 0usize, f64::NEG_INFINITY);
        for i in 0..n {
            for j in 0..n {
                if fused.agreement[[i, j]] > pv {
                    pv = fused.agreement[[i, j]];
                    pi = i;
                    pj = j;
                }
            }
        }
        assert!(
            (5..=7).contains(&pi) && (5..=7).contains(&pj),
            "fused agreement peak ({pi},{pj}) must localize the lesion (centre {centre})"
        );
        assert!(
            pv > 0.5,
            "fused agreement at the lesion must be strong, got {pv}"
        );
    }
}
