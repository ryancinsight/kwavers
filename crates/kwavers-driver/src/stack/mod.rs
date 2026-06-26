//! Physics-guided optimisation of the **multi-tile stack**: how many boards, and how many channels
//! per board, to drive a target channel count under thermal, enclosure-height, and driver-capacity
//! constraints.
//!
//! This optimises the dimension above a single board — the *number of boards*. Two physical forces
//! oppose each other:
//! * **Thermal** pushes the board count *up*: packing more channels onto one tile raises its
//!   dissipation `P = n·p_ch` and hence its steady-state rise `ΔT = P·θ` (board-to-ambient thermal
//!   resistance `θ`); spreading the channels across more tiles lowers each tile's rise.
//! * **Enclosure height** pushes the board count *down*: each added board costs one `board_pitch`
//!   of stack height (the board-to-board connector height), bounded by `height_max`.
//! * **Driver capacity** sets the floor: a tile drives at most `channel_cap` channels.
//!
//! The optimiser returns the *fewest* boards whose per-tile temperature rise stays within budget and
//! whose stack fits the enclosure — the minimal, coolest-enough, fits-in-the-box configuration.
//!
//! # Slice layout
//!
//! (Plain backticks rather than `[`X`]` intra-doc links for the slice-private submodule names —
//! private (non-`pub`) submodules would trip rustdoc's `private_intra_doc_links` lint if linked;
//! plain backticks dodge it while keeping the public types each submodule hosts clickable as
//! `[`Type`]`. Matches the established codebase precedent at
//! `cost/{adapter.rs, geometry_modulated.rs, mod.rs, physics.rs}` + `route/tree.rs`.)
//! The slice is carved by **role** (Phase 4e output-slice migration), not by file-size symmetry:
//! * `plan` — single-board thermal/height/capacity optimiser ([`StackConstraints`], [`StackPlan`],
//!   [`board_rise_k`], [`optimize_stack`]).
//! * `role` — the [`StackBoardRole`] controller/driver enum + manifest spelling.
//! * `manifest` — per-board stack-connector observation ([`StackBoardManifest`]) and its extraction
//!   from a generated board.
//! * `compatibility` — controller↔driver connector mating check ([`StackCompatibility`],
//!   [`verify_stack_pair`]).
//! * `shield` — the full controller-plus-HV shield stack ([`ShieldStackPlan`],
//!   [`ShieldStackAssembly`], [`assemble_shield_stack`], [`optimize_shield_stack`]).
//! * `util` — slice-private geometry/canonicalisation helpers shared by `manifest` + `compatibility`.

mod compatibility;
mod manifest;
mod plan;
mod role;
mod shield;
mod util;

#[cfg(test)]
mod tests;

pub use compatibility::{verify_stack_pair, StackCompatibility};
pub use manifest::{stack_board_manifest_from_board, StackBoardManifest};
pub use plan::{board_rise_k, optimize_stack, StackConstraints, StackPlan};
pub use role::StackBoardRole;
pub use shield::{
    assemble_shield_stack, optimize_shield_stack, ShieldStackAssembly, ShieldStackPlan,
    StackBoardInstance, StackTileChannelMap,
};
