//! Experiment / simulation / transient failures.
//!
//! Used by [`crate::pipeline`], [`crate::five_level`], and the eventual
//! `kwavers-transducer` acoustic simulator integration for failures that surface
//! while *running* the simulation rather than at parse / validate time. The
//! slice is small because simulation is small today — but every variant is
//! forward-looking, so the experiment layer can grow without further Error
//! churn.
//!
//! The DIP-seam escape ([`Experiment::DipSeam`]) is the typed conduit for the
//! dependency-inversion principle: when the experiment orchestrator is asked
//! for something the IO layer provably cannot provide, it returns this variant
//! rather than panicking or lying about a default value.

/// Experiment / simulation transient failure.
///
/// `Debug` only — `NonFiniteTransient::t_s` is an `f64` time, which violates `Eq`
/// (`NaN != NaN`); the other two variants (`NoTileProfile`, `DipSeam`) carry only ZSTs
/// or `&'static str` (Eq-friendly) but the mixed-field uniformity with the `f64`-bearing
/// variant forces the slice to drop `Clone + PartialEq + Eq`, matching the rest of the
/// physics-slice tree at `physics::{thermal,emi,pdn,si,acoustic}`.
/// Contrast: [`Geometry`](super::geometry::Geometry) keeps `Copy + Eq` because every
/// field is integer; `Experiment`'s mixed `f64` + ZST/`&'static str` drops them.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Experiment {
    /// A simulation step produced a non-finite pressure / voltage / temperature value;
    /// downstream pipelines cannot continue from `NaN` / `±∞`.
    #[error("simulation step {step} returned non-finite value at t={t_s} s")]
    NonFiniteTransient {
        /// Step index where the value diverged.
        step: usize,
        /// Time at the diverging step (seconds).
        t_s: f64,
    },

    /// A pulser stimulation profile references a tile profile that does not exist.
    #[error("pulser profile references a non-existent tile profile")]
    NoTileProfile,

    /// DIP seam escape — the experiment orchestrator declined to cross the
    /// dependency-inversion line. Returns when an experiment needs an IO-side
    /// capability it cannot reach (e.g. a kwavers call when the `kwavers`
    /// feature is OFF); the engine surfaces this so the user knows to enable
    /// the feature rather than silently swallowing the call.
    #[error("DIP seam: experiment cannot reach IO layer ({capability})")]
    DipSeam {
        /// What capability was requested.
        capability: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_transient_names_step_and_time() {
        let err = Experiment::NonFiniteTransient {
            step: 42,
            t_s: 1.5e-6,
        };
        let s = err.to_string();
        assert!(s.contains("42"), "step index must appear: {s}");
        assert!(
            s.contains("0.0000015") || s.contains("1.5e-6") || s.contains("1.5e-06"),
            "time must appear: {s}"
        );
    }

    #[test]
    fn dip_seam_names_the_missing_capability() {
        let err = Experiment::DipSeam {
            capability: "kwavers::acoustic::AcousticSimulator::step",
        };
        assert!(err
            .to_string()
            .contains("kwavers::acoustic::AcousticSimulator::step"));
    }

    /// Seat-belt check.
    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(e: Experiment) -> &'static str {
            match e {
                Experiment::NonFiniteTransient { .. } => "nf",
                Experiment::NoTileProfile => "ntp",
                Experiment::DipSeam { .. } => "dip",
                _ => "future-variant",
            }
        }
    }
}
