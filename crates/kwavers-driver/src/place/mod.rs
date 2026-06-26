//! Physics- and DFM-guided component placement.
//!
//! A simulated-annealing placer (`anneal`) minimises a placement energy (`energy`) that encodes
//! the manufacturing best practices: courtyard non-overlap, board-edge keep-in, **connectors to the
//! periphery / active ICs to the core**, decoupling caps next to their IC power pins, net
//! wirelength, connector ingress toward the board core, and thermal spread between power devices.
//! Output placements feed the router.
//!
//! # Module layout
//!
//! The place slice is split across several sibling files per the spec's
//! `place/{mod, anneal, energy, footprint, import, rotation, tests}.rs` layout (Phase 2c
//! completed the carve-out; the `footprint_import.rs` + `symbol_import.rs` split supersedes the
//! spec's single `import.rs` because the two import paths parse distinct grammars — the
//! `.kicad_mod` pad/footprint geometry vs the `.kicad_sym` name↔number pin map — and keeping
//! them separate gives each parser its focused tests + error envelope):
//!
//! * `anneal` — the simulated-annealing placement loop (seeded SplitMix64 PRNG, force-directed
//!   moves, footprint-policy-gated rotation moves).
//! * `energy` — the placement cost function (overlap, edge, periphery, decoupling, termination,
//!   HPWL, thermal, airflow blockage, utilization, alignment, regional, flow crossing,
//!   channel blockage, IC spread, isolation drift, mech keepout, congestion).
//! * `component` — placed component instances (`Component`, `Placement`, `Rect`,
//!   `ComponentClearanceViolation`, `component_clearance_violations`).
//! * `footprint` — footprint geometry: `Role`, `IsolationDomain`, `PadDef`, `Model3D`,
//!   `FootprintDef` with its builder methods.
//! * `footprint_import` — the `.kicad_mod` real-manufacturer-footprint importer (parser +
//!   pin-name wiring + model-recentring).
//! * `rotation` — 4-variant ZST `Rot` marker + 3-variant `RotationPolicy` (placement rotation
//!   freedom) and its `for_role` helper. Carved out of `footprint.rs` at Phase 2c.
//! * `symbol_import` — the `.kicad_sym` pin-name↔number-map importer.
//! * `tests` (gated `#[cfg(test)]`) — the 55 place-slice tests collected from the previously
//!   inline `mod tests { … }` blocks of `mod.rs`, `footprint.rs`, `footprint_import.rs`,
//!   `component.rs`, and `symbol_import.rs`. Carved out at Phase 2c.

pub mod anneal;
pub mod component;
pub mod energy;
pub mod footprint;
pub mod footprint_import;
pub mod rotation;
pub mod symbol_import;

pub use anneal::{anneal, AnnealParams};
pub use component::{
    component_clearance_violations, Component, ComponentClearanceViolation, Placement, Rect,
};
pub use energy::{energy, Axis, CongestionField, EnergyTerms, PlaceConfig, PlaceWeights};
pub use footprint::{FootprintDef, IsolationDomain, PadDef, Role};
pub use footprint_import::import_kicad_mod;
pub use rotation::{Rot, RotationPolicy};
pub use symbol_import::{import_symbol_pinmap, PinMap};

#[cfg(test)]
mod tests;
