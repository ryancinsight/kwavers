//! Acoustic / pulser-domain failures.
//!
//! Forward-looking variants for the acoustic slice (lives in
//! [`crate::physics::acoustic`] today; will be joined by the eventual
//! `kwavers-transducer` simulator integration at Phase 6). Today the slice
//! returns numeric values; Phase 2 wraps physically-implausible inputs and
//! pole-zero overflows into `Result<_, Acoustic>`.

/// Acoustic / pulser-domain failure.
///
/// `Debug` only — most variants carry `f64` pressure/MI measurements (`focal_m`/
/// `transducer_radius_m`, `mi`/`limit`/`depth_m`, `t_s`/`channel`, `lobe_db`/
/// `budget_db`); `ProfileInconsistent` carries a `String` profile-summary, and the lone
/// `usize` channel index on `NonFinitePressure` would be Eq-friendly in isolation. The
/// `f64` mix dominates — `Eq` is lost regardless, so the slice uniformly drops
/// `Clone + PartialEq + Eq`. [`Geometry`](super::super::geometry::Geometry) keeps `Copy + Eq`
/// because every field is integer; `Acoustic` drops them because of the
/// `f64`-measurement convention.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Acoustic {
    /// Focal distance is outside the array's near-field / far-field transition —
    /// physically realizable but the engineering model stops being a good fit.
    #[error(
        "focal distance {focal_m} m is outside the transducer's near-field / far-field region \
         (transducer radius {transducer_radius_m} m)"
    )]
    FocalMismatch {
        /// Requested focal distance (m).
        focal_m: f64,
        /// Transducer radius (m).
        transducer_radius_m: f64,
    },

    /// Stimulation profile does not match the transducer geometry — usually a
    /// channel-count mismatch or an aperture / focal mismatch between the
    /// driver and the resolved beam.
    #[error("pulser stimulation profile is inconsistent with transducer geometry: {0}")]
    ProfileInconsistent(String),

    /// Acoustic intensity exceeds the FDA / IEC mechanical-index safety limit.
    #[error("mechanical index {mi} exceeds safety limit {limit} at depth {depth_m:.3} m")]
    MechanicalIndexExceeded {
        /// Computed MI.
        mi: f64,
        /// Regulatory limit (typically 1.9).
        limit: f64,
        /// Depth the breach was evaluated at (m).
        depth_m: f64,
    },

    /// Acoustic pressure computation produced a NaN / ±∞ — almost always
    /// a non-physical input (frequency ≤ 0, sound_speed ≤ 0, negative attenuation).
    #[error(
        "acoustic pressure computation diverged at t={t_s} s channel={channel}: result is non-finite"
    )]
    NonFinitePressure {
        /// Time of divergence (s).
        t_s: f64,
        /// Channel index that diverged.
        channel: usize,
    },

    /// Grating-lobe level exceeds the configured spatial-discrimination budget.
    #[error("grating-lobe level {lobe_db:.1} dB exceeds budget {budget_db:.1} dB")]
    GratingLobeExceeds {
        /// Level of the worst grating lobe (dB below main lobe).
        lobe_db: f64,
        /// Allowed budget (dB).
        budget_db: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focal_mismatch_names_geometry() {
        let s = Acoustic::FocalMismatch {
            focal_m: 0.011,
            transducer_radius_m: 0.009,
        }
        .to_string();
        assert!(s.contains("0.011 m"));
        assert!(s.contains("0.009 m"));
    }

    #[test]
    fn mechanical_index_exceeded_names_depth() {
        let s = Acoustic::MechanicalIndexExceeded {
            mi: 2.4,
            limit: 1.9,
            depth_m: 0.025,
        }
        .to_string();
        assert!(s.contains("2.4"));
        assert!(s.contains("1.9"));
        assert!(s.contains("0.025 m"));
    }

    /// Seat-belt.
    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(a: Acoustic) -> &'static str {
            match a {
                Acoustic::FocalMismatch { .. } => "fm",
                Acoustic::ProfileInconsistent(_) => "pi",
                Acoustic::MechanicalIndexExceeded { .. } => "mie",
                Acoustic::NonFinitePressure { .. } => "nfp",
                Acoustic::GratingLobeExceeds { .. } => "gle",
                _ => "future-variant",
            }
        }
    }
}
