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

// ─── Other-organ SWE classification (§11.11.2–4) ────────────────────────────
//
// Unlike METAVIR liver staging, these organ tables (§11.11.2–4) are reference
// *ranges* with overlap, not independently-validated cut-offs. The classifiers
// below use the published **category-onset boundaries** (lower bound of each
// successively stiffer category) as ordered cut-offs, so a value is assigned the
// most-advanced category whose range it has entered — a conservative,
// screening-oriented reading. The same ±15 % protocol/manufacturer sensitivity
// (§11.11.1) applies, plus the organ-specific caveats noted on each function.
//
// Young's vs shear modulus: liver/prostate tables report the shear modulus `μ`,
// thyroid/breast report Young's modulus `E ≈ 3μ` (incompressible tissue,
// `E = 2μ(1+ν)`, `ν ≈ 0.5`); [`youngs_from_shear`] converts.

/// Young's modulus `E ≈ 3μ` \[same units] for nearly-incompressible tissue
/// (`E = 2μ(1+ν)`, `ν ≈ 0.5`). SWE devices report one or the other.
#[must_use]
pub fn youngs_from_shear(shear_modulus: f64) -> f64 {
    3.0 * shear_modulus
}

/// Prostate peripheral-zone SWE category (§11.11.2, Table 10.5), ordered by
/// increasing stiffness / Gleason correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProstateCategory {
    /// Benign peripheral-zone tissue.
    Benign,
    /// Prostatitis (inflammatory stiffening).
    Prostatitis,
    /// Low-grade prostate cancer (Gleason 6).
    LowGradePca,
    /// High-grade prostate cancer (Gleason 7–10).
    HighGradePca,
}

/// Prostate shear-modulus `μ` \[kPa] category-onset cut-offs (Table 10.5 lower
/// bounds: prostatitis 5, low-grade PCa 8, high-grade PCa 20).
pub const PROSTATE_SHEAR_MODULUS_CUTOFFS_KPA: [f64; 3] = [5.0, 8.0, 20.0];

/// Classify prostate SWE from shear modulus `μ` \[kPa] (§11.11.2). Confounded by
/// zonal anatomy (stiffer transition zone), capsule artefacts, and BPH; for the
/// peripheral zone. `μ ≤ 0`/non-finite → `Benign`.
#[must_use]
pub fn classify_prostate(shear_modulus_kpa: f64) -> ProstateCategory {
    let c = &PROSTATE_SHEAR_MODULUS_CUTOFFS_KPA;
    if !(shear_modulus_kpa.is_finite() && shear_modulus_kpa >= c[0]) {
        ProstateCategory::Benign
    } else if shear_modulus_kpa < c[1] {
        ProstateCategory::Prostatitis
    } else if shear_modulus_kpa < c[2] {
        ProstateCategory::LowGradePca
    } else {
        ProstateCategory::HighGradePca
    }
}

/// Thyroid-nodule malignancy risk from SWE (§11.11.3, Table 10.6), ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThyroidMalignancyRisk {
    /// Benign colloid nodule.
    Low,
    /// Follicular adenoma (low–intermediate).
    LowIntermediate,
    /// Papillary carcinoma.
    High,
    /// Anaplastic carcinoma.
    VeryHigh,
}

/// Thyroid Young's-modulus `E` \[kPa] category-onset cut-offs (Table 10.6 lower
/// bounds: follicular adenoma 15, papillary carcinoma 40, anaplastic 200).
pub const THYROID_YOUNGS_MODULUS_CUTOFFS_KPA: [f64; 3] = [15.0, 40.0, 200.0];

/// Classify thyroid-nodule malignancy risk from Young's modulus `E` \[kPa]
/// (§11.11.3). `E ≤ 0`/non-finite → `Low`.
#[must_use]
pub fn classify_thyroid(youngs_modulus_kpa: f64) -> ThyroidMalignancyRisk {
    let c = &THYROID_YOUNGS_MODULUS_CUTOFFS_KPA;
    if !(youngs_modulus_kpa.is_finite() && youngs_modulus_kpa >= c[0]) {
        ThyroidMalignancyRisk::Low
    } else if youngs_modulus_kpa < c[1] {
        ThyroidMalignancyRisk::LowIntermediate
    } else if youngs_modulus_kpa < c[2] {
        ThyroidMalignancyRisk::High
    } else {
        ThyroidMalignancyRisk::VeryHigh
    }
}

/// Breast-lesion BI-RADS upgrade likelihood from SWE (§11.11.4, Table 10.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BiradsUpgradeLikelihood {
    /// Minimal upgrade likelihood (cyst / fibroadenoma stiffness range).
    Minimal,
    /// Moderate upgrade likelihood (DCIS / soft-malignancy range).
    Moderate,
    /// High upgrade likelihood (invasive ductal carcinoma range).
    High,
}

/// Breast maximum Young's-modulus `E_max` \[kPa] category-onset cut-offs
/// (Table 10.7: DCIS-onset 30, IDC-onset 60).
pub const BREAST_YOUNGS_MAX_CUTOFFS_KPA: [f64; 2] = [30.0, 60.0];

/// Classify breast-lesion BI-RADS upgrade likelihood from `E_max` \[kPa]
/// (§11.11.4).
///
/// **Caveat:** mucinous and medullary carcinomas present as *soft* on SWE
/// (`E_max` as low as 15 kPa) and must **not** be down-classified on stiffness
/// alone — a `Minimal`/`Moderate` result does not exclude these. `E ≤ 0`/non-finite
/// → `Minimal`.
#[must_use]
pub fn classify_breast(youngs_max_kpa: f64) -> BiradsUpgradeLikelihood {
    let c = &BREAST_YOUNGS_MAX_CUTOFFS_KPA;
    if !(youngs_max_kpa.is_finite() && youngs_max_kpa >= c[0]) {
        BiradsUpgradeLikelihood::Minimal
    } else if youngs_max_kpa < c[1] {
        BiradsUpgradeLikelihood::Moderate
    } else {
        BiradsUpgradeLikelihood::High
    }
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

    /// Prostate: each category selected by a representative `μ`; half-open
    /// `[onset, next)` boundaries; monotone with stiffness; non-physical → Benign.
    #[test]
    fn classifies_prostate_categories() {
        assert_eq!(classify_prostate(3.0), ProstateCategory::Benign); // 1.5–5
        assert_eq!(classify_prostate(6.0), ProstateCategory::Prostatitis); // 5–8
        assert_eq!(classify_prostate(12.0), ProstateCategory::LowGradePca); // 8–20
        assert_eq!(classify_prostate(40.0), ProstateCategory::HighGradePca); // ≥20
        // Onsets round up to the stiffer category.
        assert_eq!(classify_prostate(5.0), ProstateCategory::Prostatitis);
        assert_eq!(classify_prostate(5.0 - 1e-9), ProstateCategory::Benign);
        assert_eq!(classify_prostate(20.0), ProstateCategory::HighGradePca);
        assert_eq!(classify_prostate(20.0 - 1e-9), ProstateCategory::LowGradePca);
        assert_eq!(classify_prostate(-1.0), ProstateCategory::Benign);
        assert_eq!(classify_prostate(f64::NAN), ProstateCategory::Benign);
        // Monotone.
        let mu = [3.0, 6.0, 12.0, 40.0];
        for w in mu.windows(2) {
            assert!(classify_prostate(w[0]) < classify_prostate(w[1]));
        }
    }

    /// Thyroid: malignancy risk rises with Young's modulus `E`; the `E ≈ 3μ`
    /// conversion places a μ-measured nodule on the correct `E` cut-off.
    #[test]
    fn classifies_thyroid_risk() {
        assert_eq!(classify_thyroid(10.0), ThyroidMalignancyRisk::Low); // colloid <15
        assert_eq!(classify_thyroid(25.0), ThyroidMalignancyRisk::LowIntermediate); // adenoma 15–40
        assert_eq!(classify_thyroid(80.0), ThyroidMalignancyRisk::High); // papillary 40–200
        assert_eq!(classify_thyroid(250.0), ThyroidMalignancyRisk::VeryHigh); // anaplastic ≥200
        assert_eq!(classify_thyroid(40.0), ThyroidMalignancyRisk::High);
        assert_eq!(classify_thyroid(40.0 - 1e-9), ThyroidMalignancyRisk::LowIntermediate);
        assert_eq!(classify_thyroid(0.0), ThyroidMalignancyRisk::Low);
        // E = 3μ: μ = 20 kPa ⇒ E = 60 kPa ⇒ High (papillary band).
        assert_eq!(
            classify_thyroid(youngs_from_shear(20.0)),
            ThyroidMalignancyRisk::High
        );
        let e = [10.0, 25.0, 80.0, 250.0];
        for w in e.windows(2) {
            assert!(classify_thyroid(w[0]) < classify_thyroid(w[1]));
        }
    }

    /// Breast: BI-RADS upgrade likelihood from `E_max`; soft-malignancy caveat
    /// means a low value is `Minimal`/`Moderate` (does not exclude mucinous).
    #[test]
    fn classifies_breast_upgrade_likelihood() {
        assert_eq!(classify_breast(15.0), BiradsUpgradeLikelihood::Minimal); // cyst/fibroadenoma <30
        assert_eq!(classify_breast(45.0), BiradsUpgradeLikelihood::Moderate); // DCIS 30–60
        assert_eq!(classify_breast(120.0), BiradsUpgradeLikelihood::High); // IDC ≥60
        assert_eq!(classify_breast(30.0), BiradsUpgradeLikelihood::Moderate);
        assert_eq!(classify_breast(30.0 - 1e-9), BiradsUpgradeLikelihood::Minimal);
        assert_eq!(classify_breast(60.0), BiradsUpgradeLikelihood::High);
        assert_eq!(classify_breast(-1.0), BiradsUpgradeLikelihood::Minimal);
        // Soft mucinous carcinoma (E_max = 15 kPa) reads Minimal by stiffness —
        // documented exception, not a contradiction.
        assert_eq!(classify_breast(15.0), BiradsUpgradeLikelihood::Minimal);
        let e = [15.0, 45.0, 120.0];
        for w in e.windows(2) {
            assert!(classify_breast(w[0]) < classify_breast(w[1]));
        }
    }

    /// `E ≈ 3μ` for incompressible tissue.
    #[test]
    fn youngs_modulus_is_triple_shear() {
        assert!((youngs_from_shear(10.0) - 30.0).abs() < 1e-12);
    }
}
