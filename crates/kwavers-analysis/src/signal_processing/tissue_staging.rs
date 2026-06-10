//! SWE-based clinical tissue staging (Chapter 11 §11.11, Algorithm 11.5).
//!
//! Maps a measured shear modulus (or a region-of-interest of moduli) to a
//! clinical stage using published shear-wave-elastography cut-offs. The leading
//! application is **liver fibrosis** scored on the METAVIR `F0…F4` scale.
//!
//! These are *threshold lookups on validated reference tables*, not a trained
//! classifier — the cut-offs carry a documented `±15 %` protocol/manufacturer
//! sensitivity (§11.11.1), so a borderline value near a boundary should be read
//! together with the heterogeneity flag from [`classify_liver_roi`].
//!
//! # References
//! - Bavu, É., et al. (2011). "Noninvasive in vivo liver fibrosis evaluation
//!   using supersonic shear imaging." *Ultrasound Med. Biol.* 37(9), 1361–1373.
//! - EFSUMB Guidelines (2013/2017) on clinical elastography.

/// METAVIR liver-fibrosis stage. Ordered by increasing severity
/// (`F0 < F1 < F2 < F3 < F4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FibrosisStage {
    /// No fibrosis.
    F0,
    /// Portal fibrosis.
    F1,
    /// Periportal fibrosis.
    F2,
    /// Bridging fibrosis.
    F3,
    /// Cirrhosis.
    F4,
}

/// METAVIR shear-modulus cut-offs `[μ kPa]` separating consecutive stages
/// (`F0|F1 = 1.7`, `F1|F2 = 2.9`, `F2|F3 = 4.8`, `F3|F4 = 9.0`); 2D-SWE,
/// fasting (Chapter 11 Table 10.4, the `μ` column).
pub const METAVIR_SHEAR_MODULUS_CUTOFFS_KPA: [f64; 4] = [1.7, 2.9, 4.8, 9.0];

/// Classify liver fibrosis from a shear modulus `μ` \[kPa] using the METAVIR
/// cut-offs. A value on a boundary is assigned to the **higher** stage
/// (intervals are `[lower, upper)`).
///
/// `μ ≤ 0` or non-finite returns `F0` (no measurable stiffening).
#[must_use]
pub fn classify_liver_fibrosis(shear_modulus_kpa: f64) -> FibrosisStage {
    if !(shear_modulus_kpa.is_finite() && shear_modulus_kpa > 0.0) {
        return FibrosisStage::F0;
    }
    let c = &METAVIR_SHEAR_MODULUS_CUTOFFS_KPA;
    if shear_modulus_kpa < c[0] {
        FibrosisStage::F0
    } else if shear_modulus_kpa < c[1] {
        FibrosisStage::F1
    } else if shear_modulus_kpa < c[2] {
        FibrosisStage::F2
    } else if shear_modulus_kpa < c[3] {
        FibrosisStage::F3
    } else {
        FibrosisStage::F4
    }
}

/// Classify liver fibrosis from a shear-wave **speed** `c_S` \[m·s⁻¹] and tissue
/// density `ρ` \[kg·m⁻³], via `μ = ρ c_S²` (then [`classify_liver_fibrosis`] in
/// kPa). Non-positive `c_S`/`ρ` returns `F0`.
#[must_use]
pub fn classify_liver_fibrosis_from_speed(shear_speed_m_s: f64, density_kg_m3: f64) -> FibrosisStage {
    if !(shear_speed_m_s.is_finite()
        && density_kg_m3.is_finite()
        && shear_speed_m_s > 0.0
        && density_kg_m3 > 0.0)
    {
        return FibrosisStage::F0;
    }
    let mu_kpa = density_kg_m3 * shear_speed_m_s * shear_speed_m_s / 1.0e3; // Pa → kPa
    classify_liver_fibrosis(mu_kpa)
}

/// Region-of-interest fibrosis staging (Algorithm 11.5): the stage of the ROI
/// **median** modulus, plus a **heterogeneity flag**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoiFibrosisStaging {
    /// Stage assigned from the ROI median modulus.
    pub stage: FibrosisStage,
    /// ROI median shear modulus \[kPa].
    pub median_kpa: f64,
    /// ROI inter-quartile range of the modulus \[kPa].
    pub iqr_kpa: f64,
    /// `true` when `IQR > 0.3·median` — the ROI is heterogeneous and the single
    /// stage label should be treated with caution (Algorithm 11.5 step 6).
    pub heterogeneous: bool,
}

/// Stage a liver ROI from its shear-modulus samples \[kPa] (Algorithm 11.5):
/// classify the median and flag heterogeneity (`IQR > 0.3·median`).
///
/// Non-finite/non-positive samples are dropped. Returns `None` if no valid
/// sample remains.
#[must_use]
pub fn classify_liver_roi(shear_modulus_kpa: &[f64]) -> Option<RoiFibrosisStaging> {
    let mut vals: Vec<f64> = shear_modulus_kpa
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let median = quantile_sorted(&vals, 0.5);
    let iqr = quantile_sorted(&vals, 0.75) - quantile_sorted(&vals, 0.25);
    Some(RoiFibrosisStaging {
        stage: classify_liver_fibrosis(median),
        median_kpa: median,
        iqr_kpa: iqr,
        heterogeneous: iqr > 0.3 * median,
    })
}

/// The `q`-quantile (`q ∈ [0, 1]`) of `sorted` (ascending) by linear
/// interpolation between order statistics.
#[inline]
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[hi].mul_add(frac, sorted[lo] * (1.0 - frac))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each METAVIR stage is selected by a representative modulus, and stages
    /// increase monotonically with stiffness.
    #[test]
    fn classifies_each_metavir_stage() {
        assert_eq!(classify_liver_fibrosis(1.0), FibrosisStage::F0);
        assert_eq!(classify_liver_fibrosis(2.0), FibrosisStage::F1);
        assert_eq!(classify_liver_fibrosis(3.5), FibrosisStage::F2);
        assert_eq!(classify_liver_fibrosis(6.0), FibrosisStage::F3);
        assert_eq!(classify_liver_fibrosis(15.0), FibrosisStage::F4);

        // Monotone: stiffer tissue never maps to a lower stage.
        let mu = [1.0, 2.0, 3.5, 6.0, 15.0];
        for w in mu.windows(2) {
            assert!(classify_liver_fibrosis(w[0]) < classify_liver_fibrosis(w[1]));
        }
    }

    /// Boundaries are half-open `[lower, upper)`: a value exactly on a cut-off
    /// goes to the higher stage; just below stays in the lower stage.
    #[test]
    fn boundary_values_round_to_higher_stage() {
        assert_eq!(classify_liver_fibrosis(1.7), FibrosisStage::F1);
        assert_eq!(classify_liver_fibrosis(1.7 - 1e-9), FibrosisStage::F0);
        assert_eq!(classify_liver_fibrosis(9.0), FibrosisStage::F4);
        assert_eq!(classify_liver_fibrosis(9.0 - 1e-9), FibrosisStage::F3);
        // Non-physical input → F0.
        assert_eq!(classify_liver_fibrosis(-1.0), FibrosisStage::F0);
        assert_eq!(classify_liver_fibrosis(f64::NAN), FibrosisStage::F0);
    }

    /// Speed-based classification: μ = ρ c_S². c_S = 2.0 m/s at ρ=1000 ⇒
    /// μ = 4.0 kPa ⇒ F2.
    #[test]
    fn classifies_from_shear_wave_speed() {
        assert_eq!(
            classify_liver_fibrosis_from_speed(2.0, 1000.0),
            FibrosisStage::F2
        );
        // c_S = 3.2 m/s ⇒ μ = 10.24 kPa ⇒ F4 (cirrhosis).
        assert_eq!(
            classify_liver_fibrosis_from_speed(3.2, 1000.0),
            FibrosisStage::F4
        );
        assert_eq!(
            classify_liver_fibrosis_from_speed(0.0, 1000.0),
            FibrosisStage::F0
        );
    }

    /// ROI staging classifies the median and flags heterogeneity when
    /// `IQR > 0.3·median`.
    #[test]
    fn roi_classifies_median_and_flags_heterogeneity() {
        // Homogeneous F2 ROI (tight spread around ~3.5 kPa).
        let homo = [3.3, 3.4, 3.5, 3.6, 3.7];
        let r = classify_liver_roi(&homo).expect("roi");
        assert_eq!(r.stage, FibrosisStage::F2);
        assert!((r.median_kpa - 3.5).abs() < 1e-9);
        assert!(!r.heterogeneous, "tight ROI must not be flagged");

        // Heterogeneous ROI: wide spread (IQR/median > 0.3).
        let hetero = [2.0, 2.5, 3.5, 7.0, 9.0];
        let h = classify_liver_roi(&hetero).expect("roi");
        assert!(h.heterogeneous, "wide ROI must be flagged: iqr={}", h.iqr_kpa);

        // Drops invalid samples; empty after filtering → None.
        let cleaned = classify_liver_roi(&[f64::NAN, -1.0, 4.0]).expect("roi");
        assert_eq!(cleaned.stage, FibrosisStage::F2);
        assert!(classify_liver_roi(&[f64::NAN, -1.0]).is_none());
        assert!(classify_liver_roi(&[]).is_none());
    }
}
