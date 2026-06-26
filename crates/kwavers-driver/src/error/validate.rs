//! End-to-end validation contract failures.
//!
//! Used by [`crate::validate`] when reconciling a generated board against the
//! [`crate::stack::StackConstraints`], [`crate::rules::DesignRules`], and the
//! `KwaversBeamStep` contract that ties the layout to the
//! `kwavers-transducer` beam-propagation pipeline. Each variant names the *named
//! invariant* that was breached; the metric and the threshold are both carried so
//! the example log line format is human-readable *and* machine-parseable.
//!
//! The corresponding Phase-0 implementation constructed its errors through
//! `Err(format!("…"))`; this module is the typed-envelope target for that
//! migration at Phase 2 (the validate.rs slice is left in place for now and only
//! the most diagnostic-rich variants are exposed so the migration path is set).

/// End-to-end validation failure.
///
/// `Debug` only — `EnergyBudgetExceeded`/`SkewExceeded`/`AmpacityDeficit` carry `f64`
/// measurements that violate `Eq` (`NaN != NaN` contradicts reflexivity); `DrcViolations`
/// carries `usize` counts (Eq-friendly) but the mixed-field uniformity with the
/// `f64`-bearing variants forces the slice to drop `Clone + PartialEq + Eq` — match the
/// variant shape for structural equality instead of relying on `Eq`.
/// Contrast: [`Geometry`](super::geometry::Geometry) keeps `Copy + Eq` because every
/// field is `usize`/`u32`; `Validate`'s `f64` measurement-pair convention drops them.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Validate {
    /// The generated layout violates the `KwaversBeamStep` contract (aperture,
    /// focal distance, channel count, frequency, sound speed, timing step
    /// all consistent with what `kwavers-transducer` expects).
    #[error("KwaversBeamStep contract violated: {0}")]
    KwaversBeamStepContract(String),

    /// The board's energy demand exceeds the budget derived from the pulser IC's
    /// thermal / current rating.
    #[error("energy budget exceeded: requested {requested} W, available {available} W")]
    EnergyBudgetExceeded {
        /// Requested total dissipation (W).
        requested: f64,
        /// Available dissipation budget (W).
        available: f64,
    },

    /// DRC-Lite violation count exceeds the engagement threshold.
    #[error("DRC-Lite violation count {count} exceeds threshold {threshold}")]
    DrcViolations {
        /// Number of hard violations found.
        count: usize,
        /// Engagement threshold.
        threshold: usize,
    },

    /// A matched-skew group exceeds its allotted length budget.
    #[error("matched-group skew {skew_mm} mm exceeds budget {budget_mm} mm")]
    SkewExceeded {
        /// Measured skew (mm).
        skew_mm: f64,
        /// Allowed budget (mm).
        budget_mm: f64,
    },

    /// A track's ampacity falls below the required current-carrying capacity.
    #[error("track ampacity deficit: {actual_a:.3} A available, {required_a:.3} A required")]
    AmpacityDeficit {
        /// Ampacity the track carries (A).
        actual_a: f64,
        /// Required minimum ampacity (A).
        required_a: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kwavers_beam_step_contract_display_includes_message() {
        let err = Validate::KwaversBeamStepContract("aperture drift > 5%".into());
        assert_eq!(
            err.to_string(),
            "KwaversBeamStep contract violated: aperture drift > 5%"
        );
    }

    #[test]
    fn energy_budget_display_carries_both_sides() {
        let err = Validate::EnergyBudgetExceeded {
            requested: 25.0,
            available: 18.0,
        };
        assert_eq!(
            err.to_string(),
            "energy budget exceeded: requested 25 W, available 18 W"
        );
    }

    #[test]
    fn drc_violations_display_uses_count_and_threshold() {
        let err = Validate::DrcViolations {
            count: 7,
            threshold: 3,
        };
        assert_eq!(
            err.to_string(),
            "DRC-Lite violation count 7 exceeds threshold 3"
        );
    }

    /// The sub-enum is `#[non_exhaustive]` — seat-belt for downstream consumers.
    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(v: Validate) -> &'static str {
            match v {
                Validate::KwaversBeamStepContract(_) => "kbs",
                Validate::EnergyBudgetExceeded { .. } => "eb",
                Validate::DrcViolations { .. } => "drc",
                Validate::SkewExceeded { .. } => "sk",
                Validate::AmpacityDeficit { .. } => "ad",
                _ => "future-variant",
            }
        }
    }
}
