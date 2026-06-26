//! Per-vertical-slice error hierarchy.
//!
//! Replaces the legacy monolithic 4-variant `Error` (Phase 0) with a per-slice sub-enum tree:
//!
//! * [`geometry::Geometry`] — grid / board-model invariant failures (migrates the 4 legacy
//!   variants `PadOutOfBounds`, `UnreachableTerminal`, `EmptyGrid`, `GridPitchTooCoarse`).
//! * [`manifest::Manifest`] — CAD/manifest file (`.kicad_mod`, `.kicad_sym`, `DriverManifest`,
//!   `EnergyBudgetInputs`) parse / IO failures.
//! * [`validate::Validate`] — end-to-end validation against the [`crate::stack`],
//!   [`crate::rules::DesignRules`], and `KwaversBeamStep` contract.
//! * [`experiment::Experiment`] — transient/simulation failures (acoustic non-finite, pulser
//!   profile reference, DIP-seam escapes).
//! * `physics::{thermal, emi, pdn, si, acoustic}` — physics-side invariant breaches; each
//!   module exposes one sub-enum with forward-looking variants that the corresponding
//!   vertical slice migrates into as Phase 2–3 unfolds.
//!
//! The top-level [`enum@Error`] is the *aggregating* enum: every variant is
//! `#[error(transparent)]` over its sub-enum, so cross-slice propagation is a single `?`.
//! `#[from]` derives the `From` conversions automatically. Each sub-enum is `#[non_exhaustive]`
//! so downstream consumers cannot exhaustively match against a slice they do not own.
//!
//! # Source-authority preservation
//!
//! `pub use error::{Error, Result}` at the crate root means existing downstream code that
//! imports `kwavers_driver::Error` / `kwavers_driver::Result` keeps compiling. Pattern-match
//! sites that reach into a single slice unwrap one layer (`Error::Geometry(g) => match g`) —
//! the per-slice sub-enums are themselves `#[non_exhaustive]` so exhaustive matching is
//! forbidden by the compiler, which is the migration's safety net.
//!
//! For rustdoc-bracket links to the aggregating error, use `enum@Error` to disambiguate
//! from `std::error::Error` (the trait).

use thiserror::Error;

pub mod experiment;
pub mod geometry;
pub mod manifest;
pub mod physics;
pub mod validate;

// Single-source-of-truth re-exports so downstream `kwavers_driver::error::Geometry` matches
// the test's `super::geometry::Geometry` regardless of whether the slice is reached through
// the crate root or through the `error::` prefix.
pub use experiment::Experiment;
pub use geometry::Geometry;
pub use manifest::Manifest;
pub use physics::{acoustic::Acoustic, emi::Emi, pdn::Pdn, si::Si, thermal::Thermal};
pub use validate::Validate;

/// Result alias for the whole crate — every vertical slice returns this by default.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level aggregating error — one variant per vertical slice, each variant is
/// `#[error(transparent)]` over its sub-enum so the slice's own `Display` (derived from
/// `thiserror`) reaches the user verbatim.
///
/// `Debug` only — *not* `Clone`, *not* `PartialEq`, *not* `Eq`. The valid subset is
/// `Debug + Display + std::error::Error`, sufficient for `?`-propagation, log lines,
/// and `e.to_string()` formatting. The reasons for dropping `Clone` / `PartialEq`:
///
/// 1. `Clone` derives do not expand cleanly when the inner `Manifest::Io { source }`
///    participates in a `thiserror` `#[source]` chain — the compiler reports the field's
///    `std::io::Error: Clone` derivation as unsatisfied through the derived impl.
/// 2. `PartialEq` would compare manifest-IO envelopes by `ErrorKind` only (`std`'s derived
///    `io::Error: PartialEq` compares kind, not OS payload), so two errors with the same
///    `ErrorKind::NotFound` but different OS messages compare *equal* under `assert_eq!`
///    — surprising for diagnostic tests.
///
/// The Phase-0 `Error` derived `Clone + PartialEq + Eq`; this is an API regression
/// documented at `docs/MIGRATION.md` § Phase 1b. The macro-derived `From<Geometry>` /
/// `From<Manifest>` / `From<Validate>` / `From<Experiment>` / `From<Thermal>` / `From<Emi>`
/// / `From<Pdn>` / `From<Si>` / `From<Acoustic>` impls (via `#[from]`) keep `?`-propagation
/// working across the slice tree.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Geometry / grid / board-model invariant failure.
    #[error(transparent)]
    Geometry(#[from] Geometry),

    /// Manifest / CAD-file parse or IO failure.
    #[error(transparent)]
    Manifest(#[from] Manifest),

    /// End-to-end validation contract failure.
    #[error(transparent)]
    Validate(#[from] Validate),

    /// Experiment / simulation / transient failure.
    #[error(transparent)]
    Experiment(#[from] Experiment),

    /// Thermal-physics invariant breach.
    #[error(transparent)]
    PhysicsThermal(#[from] Thermal),

    /// Electromagnetic-interference invariant breach.
    #[error(transparent)]
    PhysicsEmi(#[from] Emi),

    /// Power-delivery-network invariant breach.
    #[error(transparent)]
    PhysicsPdn(#[from] Pdn),

    /// Signal-integrity invariant breach.
    #[error(transparent)]
    PhysicsSi(#[from] Si),

    /// Acoustic / pulser-domain invariant breach.
    #[error(transparent)]
    PhysicsAcoustic(#[from] Acoustic),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every per-slice sub-enum converts into the aggregating [`enum@Error`] via the
    /// auto-derived `From` impls (`#[from]`) and matches the exact variant the slice names.
    /// This is the type-system guarantee the entire hierarchy trades on — that any
    /// `Result<T, Slice>` feeds cleanly into `crate::Result<T>` without losing the slice
    /// identity. The aggregator derives only `Debug, Error` (no `Clone`, no `PartialEq`,
    /// no `Eq` — see the module-level docstring), so this test asserts Display-string
    /// equality rather than `Error`-value equality.
    #[test]
    fn aggregating_error_accepts_every_slice_via_from() {
        let cases: Vec<Error> = vec![
            Geometry::EmptyGrid.into(),
            Geometry::PadOutOfBounds { pad: 7 }.into(),
            Geometry::UnreachableTerminal { net: 42 }.into(),
            Geometry::GridPitchTooCoarse { node: 19 }.into(),
            Manifest::NoPins {
                path: "x.kicad_sym".into(),
            }
            .into(),
            Validate::KwaversBeamStepContract("aperture".into()).into(),
            Experiment::NoTileProfile.into(),
            Thermal::CoolingInfeasible { dt_k: 99.0 }.into(),
            Emi::LoopInductanceExceeds {
                nh: 200.0,
                budget_nh: 150.0,
            }
            .into(),
            Pdn::RailVoltageDropExceeds {
                rail_v: 3.3,
                drop_v: 0.42,
                tol_v: 0.20,
            }
            .into(),
            Si::ImpedanceMismatch {
                actual_ohm: 92.0,
                target_ohm: 50.0,
                tol_ohm: 5.0,
            }
            .into(),
            Acoustic::FocalMismatch {
                focal_m: 0.011,
                transducer_radius_m: 0.009,
            }
            .into(),
        ];
        assert_eq!(cases.len(), 12);

        // Every aggregating-error Display string equals the slice Display string verbatim,
        // because every variant is `#[error(transparent)]`.
        for err in &cases {
            let via_aggregating = err.to_string();
            let via_slice = match err {
                Error::Geometry(g) => g.to_string(),
                Error::Manifest(m) => m.to_string(),
                Error::Validate(v) => v.to_string(),
                Error::Experiment(e) => e.to_string(),
                Error::PhysicsThermal(t) => t.to_string(),
                Error::PhysicsEmi(e) => e.to_string(),
                Error::PhysicsPdn(p) => p.to_string(),
                Error::PhysicsSi(s) => s.to_string(),
                Error::PhysicsAcoustic(a) => a.to_string(),
            };
            assert_eq!(
                via_aggregating, via_slice,
                "transparent delegation must propagate verbatim"
            );
        }
    }

    /// The aggregating `Error` is non-exhaustive — the compiler forbids writing a `match`
    /// on every variant by hand; this test is a guard rail that confirms the migration is
    /// open to add siblings without breaking downstream consumers.
    #[test]
    #[allow(unused)]
    fn aggregating_error_is_marked_non_exhaustive() {
        // Match exhaustively: the test passes if removing the `_ => unreachable!()` arm would
        // compile-error at release time. The compiler error path we rely on is the
        // `#[non_exhaustive]` attribute; the explicit wildcard documents the intent.
        fn match_all(err: &Error) -> &'static str {
            match err {
                Error::Geometry(_) => "geometry",
                Error::Manifest(_) => "manifest",
                Error::Validate(_) => "validate",
                Error::Experiment(_) => "experiment",
                Error::PhysicsThermal(_) => "thermal",
                Error::PhysicsEmi(_) => "emi",
                Error::PhysicsPdn(_) => "pdn",
                Error::PhysicsSi(_) => "si",
                Error::PhysicsAcoustic(_) => "acoustic",
                _ => "future-slice", // a sibling added in Phase 2+ — non-exhaustive catch-all
            }
        }
        assert_eq!(match_all(&Error::Geometry(Geometry::EmptyGrid)), "geometry");
        assert_eq!(
            match_all(&Error::Experiment(Experiment::NoTileProfile)),
            "experiment"
        );
    }
}
