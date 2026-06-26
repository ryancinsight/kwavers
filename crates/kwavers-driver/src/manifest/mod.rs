//! Driver-to-simulation manifest.
//!
//! The examples emit this compact text artifact next to the KiCad boards so acoustic simulations can
//! be tied to the generated hardware, not only to article constants.
//!
//! # Slice layout
//!
//! Carved by **schema role** (Phase 4f output-slice migration). Plain backticks name the
//! slice-private submodules; the public types each hosts stay clickable.
//! * `stimulation` — the acoustic protocol schema: [`StimulationProgram`] (article-class single
//!   preset) and [`TileStimulationProfile`] (per-tile PRF/SHIFT/PHASE/RAMP for the 96-channel stack).
//! * `resistor` — [`ResistorPackage`], the IPC-7351-rated SMD damping-resistor footprint enum.
//! * `driver_manifest` — [`DriverManifest`] itself: the schema, its deterministic text round-trip
//!   (`to_text`/`from_text`/`read`), and the protocol-load accessors.
//! * `energy_budget` — [`EnergyBudgetInputs`]/[`EnergyBudgetReport`] and the
//!   [`DriverManifest::validate_v2_energy_budget`] routed-board ampacity/dissipation validator.
//! * `extract` — [`hv_manifest_from_board`], which builds the HV-board portion from a placed/routed
//!   design.
//!
//! The schema-key constants (`MANIFEST_FORMAT_V1/V2`, `MANIFEST_ARTICLE_*`) and the 96-channel lane
//! binding (`TX_LANES_V2`/`CHANNELS_PER_TILE_V2`) are the single source of truth in
//! [`crate::ssot`]; the sub-files import them from there, never re-declaring them.

mod driver_manifest;
mod energy_budget;
mod extract;
mod resistor;
mod stimulation;

#[cfg(test)]
mod tests;

pub use driver_manifest::DriverManifest;
pub use energy_budget::{EnergyBudgetInputs, EnergyBudgetReport};
pub use extract::hv_manifest_from_board;
pub use resistor::ResistorPackage;
pub use stimulation::{StimulationProgram, TileStimulationProfile};
