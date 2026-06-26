//! Signal-integrity (SI) failures.
//!
//! Forward-looking variants for the [`crate::physics::si`] slice. Today the slice
//! returns impedance / delay values directly; Phase 2 wraps breaches into
//! `Result<_, Si>`.

/// Signal-integrity failure.
///
/// `Debug` only — every variant except `CrosstalkExceeded` carries `f64` measurements
/// (`actual_ohm`/`target_ohm`/`tol_ohm`, `delta_mm`/`budget_mm`,
/// `coupling_db`/`budget_db`, `ps_per_m`/`budget_ps_per_m`), which violate `Eq`
/// (`NaN != NaN`); `CrosstalkExceeded::victim` is a `String` log-alignment tag, not a
/// comparison key. [`Geometry`](super::super::geometry::Geometry) keeps `Copy + Eq` because
/// every field is integer; `Si` drops them because of the `f64` measurement
/// convention. The lone `String` does not move the needle — `Eq` was already lost to the
/// `f64` mix.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Si {
    /// Microstrip / stripline impedance mismatch exceeds the matched-net tolerance.
    #[error(
        "impedance at node is {actual_ohm:.2} Ω, target {target_ohm:.2} Ω \
         (tolerance {tol_ohm:.2} Ω)"
    )]
    ImpedanceMismatch {
        /// Measured impedance (Ω).
        actual_ohm: f64,
        /// Target impedance (Ω).
        target_ohm: f64,
        /// Engineering tolerance (Ω).
        tol_ohm: f64,
    },

    /// Within a matched-length group, the longest-to-shortest length delta exceeds
    /// the budget derived from the bit-period / signal-rise-time.
    #[error("matched-group length mismatch {delta_mm:.3} mm exceeds budget {budget_mm:.3} mm")]
    LengthMismatch {
        /// Measured delta (mm).
        delta_mm: f64,
        /// Allowed budget (mm).
        budget_mm: f64,
    },

    /// Near-end / far-end crosstalk coupling exceeds the receiver's eye-budget.
    #[error(
        "NEXT/FEXT coupling {coupling_db:.2} dB exceeds budget {budget_db:.2} dB \
         for victim net {victim}"
    )]
    CrosstalkExceeded {
        /// Measured coupling (dB).
        coupling_db: f64,
        /// Allowed coupling budget (dB).
        budget_db: f64,
        /// Victim-net name (for log alignment).
        victim: String,
    },

    /// Rise-time degradation over the routed length exceeds the receiver budget.
    #[error("rise-time degradation {ps_per_m:.0} ps/m exceeds budget {budget_ps_per_m:.0} ps/m")]
    RiseTimeDegraded {
        /// Measured degradation (ps/m).
        ps_per_m: f64,
        /// Allowed budget (ps/m).
        budget_ps_per_m: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impedance_mismatch_display_names_all_three() {
        let s = Si::ImpedanceMismatch {
            actual_ohm: 92.0,
            target_ohm: 50.0,
            tol_ohm: 5.0,
        }
        .to_string();
        assert!(s.contains("92.00"));
        assert!(s.contains("50.00"));
        assert!(s.contains("5.00"));
    }

    #[test]
    fn length_mismatch_uses_mm_units() {
        let s = Si::LengthMismatch {
            delta_mm: 1.234,
            budget_mm: 0.500,
        }
        .to_string();
        assert!(s.contains("1.234 mm"));
        assert!(s.contains("0.500 mm"));
    }

    /// Seat-belt.
    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(s: Si) -> &'static str {
            match s {
                Si::ImpedanceMismatch { .. } => "im",
                Si::LengthMismatch { .. } => "lm",
                Si::CrosstalkExceeded { .. } => "ce",
                Si::RiseTimeDegraded { .. } => "rtd",
                _ => "future-variant",
            }
        }
    }
}
