//! Adversarial DFM / physics critic — `kwavers` placement / routing auditor.
//!
//! Implementation of the "attack axes" the optimiser uses to drive decisions
//! along manufacturing and physical risk lines. Each axis lives in its own
//! sub-module so callers can `use crate::audit::<axis>::<symbol>` directly.
//! The [`audit`] entrypoint inside `critic` combines all five axes into a
//! [`FaultReport`].
//!
//! # Slice layout (Phase 4c output-slice migration)
//!
//! (Plain backticks rather than `[X]`-escaped intra-doc link placeholders because each sub-module
//! is `mod` (private) within the slice; the public types each sub-module hosts
//! — [`FaultReport`], [`audit`], [`emi_hotspots`], [`copper_area_per_layer`],
//! [`copper_imbalance`], [`weakness_field`], [`ChargeRecyclingReport`],
//! [`charge_recycling_efficiency_audit`], [`PulseSkipInterferenceReport`],
//! [`pulse_skip_interference_audit`] — stay clickable through the `pub use`
//! block below. Matches the `cost/*.rs` + `route/tree.rs` precedent set in
//! the Phase 3 hygiene pass.)
//!
//! # Attack axes
//!
//! 1. **Lane crossings**: topological crossings requiring layer changes.
//! 2. **Clearance**: manufacturing rule violations.
//! 3. **Near-short / fault risk**: graded margins, especially across HV↔LV.
//! 4. **Crosstalk**: capacitive/inductive coupling from parallel adjacent runs.
//! 5. **Antenna / dangling**: etch/ESD risks from unconnected track ends.

#![allow(missing_docs)]  // Phase 4c carve: `pub mod X;` declarations are slice-private; lint allow covers the entire facade.

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

#[allow(missing_docs)]  // Phase 4c carve: re-exports carry docs from their source axes.
pub use fault_report::FaultReport;
pub use crosstalk::emi_hotspots;
pub use antenna::{copper_area_per_layer, copper_imbalance};
pub use critic::{
    ChargeRecyclingReport,
    rasterize_hotspots,
    rasterize_hotspots_radius,
    PulseSkipInterferenceReport,
    audit,
    charge_recycling_efficiency_audit,
    pulse_skip_interference_audit,
    weakness_field,
};
