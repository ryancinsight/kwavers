//! Whole-design **physics validation**: aggregate the per-domain models (ampacity, dielectric,
//! thermal, manufacturability) into a single pass/fail report with quantified margins, so a design
//! is signed off against analytical limits rather than eyeballed from a dashboard. Each [`Check`]
//! carries its measured value, its limit, and the signed headroom; [`PhysicsReport::all_pass`] is the
//! gate.
//!
//! # Slice layout
//!
//! Carved by **role** (Phase 4j). Plain backticks name the slice-private submodules; the public
//! items each hosts stay clickable.
//! * `check` — the [`Check`] / [`PhysicsReport`] primitives (measured value vs limit + signed margin).
//! * `board_checks` — board-geometry physics checks: HV creepage ([`min_hv_spacing_mm`]), ampacity
//!   headroom ([`worst_ampacity_margin_mm`]), the shared [`core_checks`], the [`ViaCensus`] /
//!   [`via_census`] HDI tally, [`microvia_aspect_check`], and net length / [`group_skew_mm`].
//! * `kwavers_beam` — the driver→transducer beam-validation seam: the typed [`KwaversBeamStep`] the
//!   `kwavers-transducer` simulator consumes, [`manifest_to_kwavers_beam_step`], the
//!   [`KwaversBeamValidation`] prediction set, and [`validate_against_budget`].
//!
//! Kwavers safety bounds, `Check` names, water Z₀, and SI-prefix scalars are the single source of
//! truth in [`crate::ssot`]; the sub-files import them from there.

mod board_checks;
mod check;
mod kwavers_beam;

#[cfg(test)]
mod tests;

pub use board_checks::{
    core_checks, group_skew_mm, microvia_aspect_check, min_hv_spacing_mm, net_length_mm,
    via_census, worst_ampacity_margin_mm, ViaCensus,
};
pub use check::{Check, PhysicsReport};
pub use kwavers_beam::{
    manifest_to_kwavers_beam_step, validate_against_budget, KwaversBeamStep, KwaversBeamValidation,
};
