//! Histotripsy / cavitation mechanical tissue characterization.
//!
//! Acoustic and thermal tissue properties live in [`super::tissue`]; this module
//! adds the *mechanical* and *cavitation-threshold* properties a histotripsy
//! treatment plan needs:
//! * `tensile_yield_stress` — the tissue strength `σ_y` in the cavitation lesion
//!   energy balance `R_L = R₀·(P₀·ICD/σ_y)^(1/3)` (Maxwell 2011; Vlaisavljevich
//!   2014);
//! * the intrinsic-threshold law `p_T(f) = p_T(1 MHz) + slope·log₁₀(f/1 MHz)`
//!   (Maxwell 2013; Vlaisavljevich 2015) with its Gaussian width `σ_T`.
//!
//! These are the single source of truth for the histotripsy tissue constants;
//! downstream crates (analytical book layer, Python bindings) read them here
//! rather than hard-coding values.

use super::tissue::AbsorptionTissueType;

/// Mechanical / cavitation-threshold characterization of a tissue for
/// histotripsy treatment planning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistotripsyTissueProperties {
    /// Tensile yield (failure) stress `σ_y` [Pa] used in the lesion energy
    /// balance. Soft tissues: ~1–4 kPa (Vlaisavljevich 2014).
    pub tensile_yield_stress_pa: f64,
    /// Mean intrinsic cavitation threshold peak-negative pressure at 1 MHz [Pa]
    /// (Maxwell 2013 erf-CDF model). Water-rich soft tissue: ~28 MPa.
    pub intrinsic_threshold_1mhz_pa: f64,
    /// Log-linear frequency slope of the intrinsic threshold [Pa per decade]
    /// (Vlaisavljevich 2015). Liver: ~1.4 MPa/decade over 0.25–3 MHz.
    pub threshold_slope_pa_per_decade: f64,
    /// Gaussian width `σ_T` of the single-pulse threshold distribution [Pa].
    pub threshold_sigma_pa: f64,
}

impl HistotripsyTissueProperties {
    /// Intrinsic threshold at an arbitrary frequency via the log-linear law.
    #[must_use]
    pub fn intrinsic_threshold_at(&self, freq_hz: f64) -> f64 {
        const F_REF: f64 = 1.0e6;
        let f = freq_hz.max(f64::MIN_POSITIVE);
        self.intrinsic_threshold_1mhz_pa + self.threshold_slope_pa_per_decade * (f / F_REF).log10()
    }
}

/// Generic water-rich soft-tissue characterization (the fallback / default).
/// The intrinsic cavitation threshold is set by water content, so it is similar
/// across soft tissues; the yield stress (stiffness) is what varies most.
const SOFT_TISSUE: HistotripsyTissueProperties = HistotripsyTissueProperties {
    tensile_yield_stress_pa: 2.5e3,
    intrinsic_threshold_1mhz_pa: 28.0e6,
    threshold_slope_pa_per_decade: 1.4e6,
    threshold_sigma_pa: 1.0e6,
};

/// Histotripsy mechanical / threshold characterization for a tissue type.
///
/// Sources: Maxwell et al. (2013) *Ultrasound Med. Biol.* 39, 449 (threshold,
/// width); Vlaisavljevich et al. (2014, 2015) (yield stress, frequency slope).
/// Tissues without a measured value fall back to generic soft tissue.
#[must_use]
pub fn histotripsy_tissue_properties(t: AbsorptionTissueType) -> HistotripsyTissueProperties {
    use AbsorptionTissueType as T;
    match t {
        T::Liver => HistotripsyTissueProperties {
            tensile_yield_stress_pa: 2.0e3,
            intrinsic_threshold_1mhz_pa: 28.2e6,
            threshold_slope_pa_per_decade: 1.4e6,
            threshold_sigma_pa: 0.96e6,
        },
        T::Kidney => HistotripsyTissueProperties {
            tensile_yield_stress_pa: 3.0e3,
            intrinsic_threshold_1mhz_pa: 28.0e6,
            threshold_slope_pa_per_decade: 1.4e6,
            threshold_sigma_pa: 1.0e6,
        },
        T::Brain => HistotripsyTissueProperties {
            tensile_yield_stress_pa: 1.0e3,
            intrinsic_threshold_1mhz_pa: 26.0e6,
            threshold_slope_pa_per_decade: 1.4e6,
            threshold_sigma_pa: 1.0e6,
        },
        T::Muscle => HistotripsyTissueProperties {
            tensile_yield_stress_pa: 4.0e3,
            intrinsic_threshold_1mhz_pa: 28.0e6,
            threshold_slope_pa_per_decade: 1.4e6,
            threshold_sigma_pa: 1.0e6,
        },
        // Higher-fat / lower-water tissues cavitate at a higher threshold.
        T::Fat | T::BreastFat => HistotripsyTissueProperties {
            tensile_yield_stress_pa: 1.5e3,
            intrinsic_threshold_1mhz_pa: 30.0e6,
            threshold_slope_pa_per_decade: 1.4e6,
            threshold_sigma_pa: 1.2e6,
        },
        // Pure water / gel phantoms: the canonical intrinsic threshold, ~no strength.
        T::Water => HistotripsyTissueProperties {
            tensile_yield_stress_pa: 1.0e2,
            intrinsic_threshold_1mhz_pa: 28.2e6,
            threshold_slope_pa_per_decade: 1.4e6,
            threshold_sigma_pa: 0.96e6,
        },
        _ => SOFT_TISSUE,
    }
}

/// Map a tissue name (case-insensitive) to its type, then return its histotripsy
/// characterization. Unknown names fall back to generic soft tissue.
#[must_use]
pub fn histotripsy_tissue_properties_by_name(name: &str) -> HistotripsyTissueProperties {
    use AbsorptionTissueType as T;
    let t = match name.to_ascii_lowercase().as_str() {
        "liver" => T::Liver,
        "kidney" => T::Kidney,
        "brain" => T::Brain,
        "muscle" => T::Muscle,
        "fat" => T::Fat,
        "breastfat" | "breast_fat" => T::BreastFat,
        "water" => T::Water,
        _ => T::SoftTissue,
    };
    histotripsy_tissue_properties(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn liver_matches_maxwell_vlaisavljevich() {
        let p = histotripsy_tissue_properties(AbsorptionTissueType::Liver);
        assert!((p.intrinsic_threshold_1mhz_pa - 28.2e6).abs() < 1.0);
        assert!((p.tensile_yield_stress_pa - 2.0e3).abs() < 1.0);
        // By-name route agrees.
        let q = histotripsy_tissue_properties_by_name("Liver");
        assert_eq!(p, q);
    }

    #[test]
    fn threshold_frequency_law_is_log_linear() {
        let p = histotripsy_tissue_properties(AbsorptionTissueType::Liver);
        // At 1 MHz the law returns the reference value.
        assert!((p.intrinsic_threshold_at(1.0e6) - p.intrinsic_threshold_1mhz_pa).abs() < 1.0);
        // One decade up adds exactly the slope.
        let up = p.intrinsic_threshold_at(10.0e6) - p.intrinsic_threshold_1mhz_pa;
        assert!((up - p.threshold_slope_pa_per_decade).abs() < 1.0);
        // Higher frequency → higher threshold (water-rich tissue).
        assert!(p.intrinsic_threshold_at(3.0e6) > p.intrinsic_threshold_at(0.5e6));
    }

    #[test]
    fn unknown_name_falls_back_to_soft_tissue() {
        assert_eq!(
            histotripsy_tissue_properties_by_name("unobtanium"),
            SOFT_TISSUE
        );
    }

    #[test]
    fn fat_threshold_exceeds_liver() {
        // Lower water content raises the cavitation threshold.
        let fat = histotripsy_tissue_properties(AbsorptionTissueType::Fat);
        let liver = histotripsy_tissue_properties(AbsorptionTissueType::Liver);
        assert!(fat.intrinsic_threshold_1mhz_pa > liver.intrinsic_threshold_1mhz_pa);
    }
}
