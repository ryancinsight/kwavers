//! Adversarial DFM / physics critic — `kwavers` placement / routing auditor.
//!
//! Implementation of the "attack axes" the optimiser uses to drive decisions
//! along manufacturing and physical risk lines. Each axis lives in its own
//! sub-module so callers can `use crate::audit::<axis>::<symbol>` directly.
//! The [`audit`] entrypoint inside `critic` combines all five axes into a
//! [`FaultReport`].
//!
//! # Slice layout
//!
//! (Plain backticks rather than `[X]`-escaped intra-doc link placeholders because each sub-module
//! is `mod` (private) within the slice; the public types each sub-module hosts
//! — [`FaultReport`], [`audit`], [`emi_hotspots`], [`copper_area_per_layer`],
//! [`copper_imbalance`], [`weakness_field`], [`ChargeRecyclingReport`],
//! [`charge_recycling_efficiency_audit`], [`PulseSkipInterferenceReport`],
//! [`pulse_skip_interference_audit`] — stay clickable through the `pub use`
//! block below. Matches the `cost/*.rs` + `route/tree.rs` precedent set in
//! the hygiene pass.)
//!
//! # Attack axes
//!
//! 1. **Lane crossings**: topological crossings requiring layer changes.
//! 2. **Clearance**: manufacturing rule violations.
//! 3. **Near-short / fault risk**: graded margins, especially across HV↔LV.
//! 4. **Crosstalk**: capacitive/inductive coupling from parallel adjacent runs.
//! 5. **Antenna / dangling**: etch/ESD risks from unconnected track ends.

// Blanket allow: several slice submodules below carry no module docs yet. Tracked as
// KW-DOC-110 in backlog.md, which documents them and removes this.
#![allow(missing_docs)]

pub mod antenna;
pub mod critic;
pub mod crosstalk;
pub mod detect_diff_pair;
pub mod detect_high_speed;
pub mod detect_power;
pub mod detect_track;
pub mod fault_report;
pub mod net_util;
pub mod shorts;

/// Integration tests — `#[cfg(test)]` so the test surface stays out of the
/// production binary. Matches the `crate::cost::tests` precedent.
#[cfg(test)]
mod tests;

pub use antenna::{copper_area_per_layer, copper_imbalance};
pub use critic::{
    audit, charge_recycling_efficiency_audit, pulse_skip_interference_audit, rasterize_hotspots,
    rasterize_hotspots_radius, weakness_field, ChargeRecyclingReport, PulseSkipInterferenceReport,
};
pub use crosstalk::emi_hotspots;
pub use fault_report::FaultReport;
