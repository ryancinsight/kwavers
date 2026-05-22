use crate::core::constants::REFERENCE_FREQUENCY_HZ;
/// Baseline intrinsic-threshold magnitude at 1 MHz in water-rich soft tissue
/// (Maxwell 2013, Table II — bovine liver).
const PT0_PA: f64 = 28.2e6;
/// Slope of the log-frequency fit (Vlaisavljevich 2015, Fig. 6 fit to liver).
const PT_SLOPE_PA_PER_DECADE: f64 = 1.4e6;

/// Qualitative benefit/detriment summary for a clinical regime.
///
/// Values are short human-readable strings sourced from the cited
/// literature; they are intended for clinical decision-support displays
/// and for chapter-level documentation generation.
#[derive(Debug, Clone, Copy)]
pub struct BenefitDetriment {
    pub benefits: &'static [&'static str],
    pub detriments: &'static [&'static str],
}

/// Intrinsic-threshold pressure magnitude `p_t(f)` for water-rich soft
/// tissue at frequency `f` (Hz), in pascals. Returns a positive number;
/// callers comparing with peak-negative pressure should use
/// `|p^-_min| >= intrinsic_threshold_pa(f)`.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[must_use]
pub fn intrinsic_threshold_pa(frequency_hz: f64) -> f64 {
    debug_assert!(frequency_hz > 0.0);
    let log_ratio = (frequency_hz / REFERENCE_FREQUENCY_HZ).log10();
    PT_SLOPE_PA_PER_DECADE.mul_add(log_ratio, PT0_PA)
}
