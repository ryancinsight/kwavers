//! Image-quality metrics for ultrasound / photoacoustic reconstructions
//! (Chapter 9 §9.8).
//!
//! Contrast metrics compare a lesion pixel population `l` against a background
//! population `b` (Definition 9.10):
//! - [`contrast_ratio_db`] — `CR = 20·log₁₀(μ_l / μ_b)` \[dB].
//! - [`contrast_to_noise_ratio`] — `CNR = |μ_l − μ_b| / √(σ_l² + σ_b²)`.
//! - [`generalized_cnr`] — `gCNR = 1 − OVL`, where `OVL` is the histogram
//!   overlap of the two intensity distributions (Rodriguez-Molares et al.,
//!   2020). Per Theorem 9.9, gCNR is invariant under any monotone-increasing
//!   image transform (log compression, TGC); CNR and CR are not.
//!
//! [`fwhm`] measures the full-width at half-maximum of a 1-D PSF profile
//! (Definition 9.8) for lateral/axial resolution.
//!
//! # References
//! - Rodriguez-Molares, A., et al. (2020). "The Generalized Contrast-to-Noise
//!   Ratio: A Formal Definition for Lesion Detectability." *IEEE Trans.
//!   Ultrason. Ferroelectr. Freq. Control* 67(4), 745–759.

/// Mean and population standard deviation of the finite entries of `data`.
/// Returns `None` when no finite sample remains.
fn mean_std(data: &[f64]) -> Option<(f64, f64)> {
    let vals: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return None;
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    Some((mean, var.sqrt()))
}

/// Contrast ratio `CR = 20·log₁₀(μ_l / μ_b)` \[dB] (Definition 9.10).
///
/// Requires strictly-positive lesion and background means (intensities, not
/// log-compressed values); returns `None` otherwise.
#[must_use]
pub fn contrast_ratio_db(lesion: &[f64], background: &[f64]) -> Option<f64> {
    let (mu_l, _) = mean_std(lesion)?;
    let (mu_b, _) = mean_std(background)?;
    if mu_l > 0.0 && mu_b > 0.0 {
        Some(20.0 * (mu_l / mu_b).log10())
    } else {
        None
    }
}

/// Contrast-to-noise ratio `CNR = |μ_l − μ_b| / √(σ_l² + σ_b²)` (Definition 9.10).
///
/// Returns `None` if either population is empty or both populations have zero
/// variance (undefined noise floor).
#[must_use]
pub fn contrast_to_noise_ratio(lesion: &[f64], background: &[f64]) -> Option<f64> {
    let (mu_l, sd_l) = mean_std(lesion)?;
    let (mu_b, sd_b) = mean_std(background)?;
    let noise = (sd_l * sd_l + sd_b * sd_b).sqrt();
    if noise > 0.0 {
        Some((mu_l - mu_b).abs() / noise)
    } else {
        None
    }
}

/// Generalized contrast-to-noise ratio `gCNR = 1 − OVL ∈ [0, 1]` (Definition
/// 9.10), where `OVL = Σ_bins min(p_l, p_b)` is the overlap of the lesion and
/// background intensity histograms (`n_bins` equal-width bins over the pooled
/// data range).
///
/// `gCNR = 0` ⇔ the distributions coincide (no detectability); `gCNR = 1` ⇔ they
/// are disjoint (perfect separation). Because the bins span the pooled range,
/// gCNR is invariant under affine rescaling exactly and under any monotone
/// transform in the fine-bin limit (Theorem 9.9), unlike CNR/CR.
///
/// Returns `0.0` (maximal overlap) for empty inputs, `n_bins == 0`, or a
/// degenerate single-valued pooled range.
#[must_use]
pub fn generalized_cnr(lesion: &[f64], background: &[f64], n_bins: usize) -> f64 {
    let l: Vec<f64> = lesion.iter().copied().filter(|v| v.is_finite()).collect();
    let b: Vec<f64> = background
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if l.is_empty() || b.is_empty() || n_bins == 0 {
        return 0.0;
    }
    let (lo, hi) = l
        .iter()
        .chain(b.iter())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if hi <= lo {
        // All pooled values equal ⇒ identical single-bin distributions.
        return 0.0;
    }
    let width = (hi - lo) / n_bins as f64;
    let bin = |v: f64| -> usize { (((v - lo) / width) as usize).min(n_bins - 1) };

    let mut hist_l = vec![0.0_f64; n_bins];
    let mut hist_b = vec![0.0_f64; n_bins];
    for &v in &l {
        hist_l[bin(v)] += 1.0;
    }
    for &v in &b {
        hist_b[bin(v)] += 1.0;
    }
    let (nl, nb) = (l.len() as f64, b.len() as f64);
    let ovl: f64 = (0..n_bins)
        .map(|i| (hist_l[i] / nl).min(hist_b[i] / nb))
        .sum();
    (1.0 - ovl).clamp(0.0, 1.0)
}

/// Full-width at half-maximum of a 1-D profile (Definition 9.8), in the units of
/// `spacing` (the inter-sample distance).
///
/// The half-maximum level is `max(profile) / 2`; the width is the distance
/// between the first up-crossing and last down-crossing of that level, located
/// by linear interpolation. Returns `None` when the profile is too short, the
/// peak is non-positive, or the half-level is not crossed on both sides.
#[must_use]
pub fn fwhm(profile: &[f64], spacing: f64) -> Option<f64> {
    if profile.len() < 2 || !spacing.is_finite() || spacing <= 0.0 {
        return None;
    }
    let max = profile.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() || max <= 0.0 {
        return None;
    }
    let half = 0.5 * max;

    let mut left = None;
    for i in 0..profile.len() - 1 {
        if profile[i] < half && profile[i + 1] >= half {
            let t = (half - profile[i]) / (profile[i + 1] - profile[i]);
            left = Some(i as f64 + t);
            break;
        }
    }
    let mut right = None;
    for i in (0..profile.len() - 1).rev() {
        if profile[i + 1] < half && profile[i] >= half {
            let t = (half - profile[i + 1]) / (profile[i] - profile[i + 1]);
            right = Some((i + 1) as f64 - t);
            break;
        }
    }
    match (left, right) {
        (Some(l), Some(r)) if r > l => Some((r - l) * spacing),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CR = 20·log₁₀(μ_l/μ_b): means 2.0 vs 1.0 ⇒ 20·log₁₀2 ≈ 6.0206 dB.
    #[test]
    fn contrast_ratio_matches_definition() {
        let lesion = [1.9, 2.0, 2.1];
        let background = [0.9, 1.0, 1.1];
        let cr = contrast_ratio_db(&lesion, &background).expect("cr");
        assert!((cr - 20.0 * 2.0_f64.log10()).abs() < 1e-9, "cr = {cr}");
        // Non-positive mean ⇒ None.
        assert!(contrast_ratio_db(&[0.0, 0.0], &background).is_none());
        assert!(contrast_ratio_db(&[], &background).is_none());
    }

    /// CNR = |μ_l − μ_b| / √(σ_l² + σ_b²) against a hand-computed value.
    #[test]
    fn cnr_matches_definition() {
        let lesion = [3.0, 5.0]; // μ=4, σ=1
        let background = [0.0, 2.0]; // μ=1, σ=1
        let cnr = contrast_to_noise_ratio(&lesion, &background).expect("cnr");
        // |4−1| / √(1+1) = 3/√2.
        assert!((cnr - 3.0 / 2.0_f64.sqrt()).abs() < 1e-12, "cnr = {cnr}");
        // Zero variance on both sides ⇒ undefined noise ⇒ None.
        assert!(contrast_to_noise_ratio(&[2.0, 2.0], &[1.0, 1.0]).is_none());
    }

    /// gCNR bounds: identical distributions ⇒ 0; disjoint ⇒ 1; symmetric.
    #[test]
    fn gcnr_bounds_and_symmetry() {
        let a = [1.0, 1.5, 2.0, 2.5, 3.0];
        // Identical populations fully overlap ⇒ gCNR = 0.
        assert!(generalized_cnr(&a, &a, 32).abs() < 1e-12);
        // Disjoint ranges ⇒ no overlap ⇒ gCNR = 1.
        let lo = [0.0, 0.1, 0.2, 0.3];
        let hi = [10.0, 10.1, 10.2, 10.3];
        assert!((generalized_cnr(&lo, &hi, 32) - 1.0).abs() < 1e-12);
        // Symmetric in its arguments.
        assert!((generalized_cnr(&lo, &hi, 16) - generalized_cnr(&hi, &lo, 16)).abs() < 1e-12);
        // Degenerate / empty guards.
        assert!(generalized_cnr(&[5.0, 5.0], &[5.0, 5.0], 8).abs() < 1e-12);
        assert!(generalized_cnr(&[], &hi, 8).abs() < 1e-12);
    }

    /// Theorem 9.9: gCNR is invariant under a monotone transform while CNR is
    /// not. Affine rescaling is exactly invariant (shared pooled-range bins);
    /// a nonlinear monotone map (square, on positive data) is invariant to
    /// within the binning error — yet CNR changes substantially.
    #[test]
    fn gcnr_invariant_under_monotone_transform_unlike_cnr() {
        let lesion = [2.0, 2.4, 2.8, 3.2, 3.6, 4.0];
        let background = [1.0, 1.3, 1.6, 1.9, 2.2, 2.5];
        let g0 = generalized_cnr(&lesion, &background, 64);

        // Exact affine invariance: x → 3x + 5.
        let aff = |v: &[f64]| v.iter().map(|x| 3.0 * x + 5.0).collect::<Vec<_>>();
        let g_aff = generalized_cnr(&aff(&lesion), &aff(&background), 64);
        assert!(
            (g0 - g_aff).abs() < 1e-12,
            "gCNR not affine-invariant: {g0} vs {g_aff}"
        );

        // Nonlinear monotone map x → x² (positive data): gCNR stays invariant to
        // within the histogram-binning error…
        let sq = |v: &[f64]| v.iter().map(|x| x * x).collect::<Vec<_>>();
        let g_sq = generalized_cnr(&sq(&lesion), &sq(&background), 64);
        assert!(
            (g0 - g_sq).abs() < 0.05,
            "gCNR drifted under x²: {g0} vs {g_sq}"
        );

        // …whereas CNR is sensitive to the same nonlinear map: it shifts by a
        // few percent, confirming it is *not* transform-invariant (Theorem 9.9).
        let cnr0 = contrast_to_noise_ratio(&lesion, &background).expect("cnr");
        let cnr_sq = contrast_to_noise_ratio(&sq(&lesion), &sq(&background)).expect("cnr");
        assert!(
            (cnr0 - cnr_sq).abs() / cnr0 > 0.03,
            "CNR should not be transform-invariant: {cnr0} vs {cnr_sq}"
        );
    }

    /// FWHM of a symmetric triangular profile: peak at index 2, half-max at
    /// indices 1 and 3 ⇒ width 2 samples × spacing.
    #[test]
    fn fwhm_triangular_profile() {
        let profile = [0.0, 0.5, 1.0, 0.5, 0.0];
        let w = fwhm(&profile, 0.25).expect("fwhm");
        assert!((w - 2.0 * 0.25).abs() < 1e-12, "fwhm = {w}");
        // Linear-interp half-crossings: peak 4.0 at index 1, half=2.0.
        let p2 = [0.0, 4.0, 0.0];
        // half=2.0 crossed at i=0 (0→4 at t=0.5 ⇒ 0.5) and falling at i=1
        // (4→0 at 0.5 ⇒ 1.5) ⇒ width 1.0 sample.
        let w2 = fwhm(&p2, 1.0).expect("fwhm");
        assert!((w2 - 1.0).abs() < 1e-12, "fwhm = {w2}");
        // Degenerate guards.
        assert!(fwhm(&[0.0, 0.0, 0.0], 1.0).is_none());
        assert!(fwhm(&[1.0], 1.0).is_none());
        assert!(fwhm(&profile, 0.0).is_none());
    }
}
