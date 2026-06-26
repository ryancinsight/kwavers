//! Automatic, self-contained verification suite run as part of layout + routing.
//!
//! The engine does not rely solely on an external tool to find problems — it verifies its own
//! output across the standard PCB sign-off axes, with `kicad-cli` (DRC/ERC) kept as an independent
//! cross-check oracle in the examples:
//!
//! * **ERC** — logical/electrical integrity over the *netlist*: pads with no net, floating
//!   (single-terminal) nets, and power pins not landed on a power/ground rail. ([`erc()`])
//! * **DRC** — physical/manufacturing rules over the *routed copper*: clearance violations,
//!   via-adjacency, acid traps, dangling ends. Provided by [`crate::audit::audit`] and folded in here.
//! * **Assembly** — component courtyard spacing, so rendered package intersections and pick-and-place
//!   clearance failures are hard manufacturing faults, not visual warnings.
//! * **LVS** — layout-versus-schematic: extract as-built connectivity from the copper (tracks, vias,
//!   pads, plane pours) and compare it to the intended netlist, reporting **opens** (a net broken
//!   into islands — i.e. an incomplete route) and **shorts** (two nets fused by copper). ([`lvs()`])
//! * **SI/PI** — signal/power integrity: worst power-net IR drop ([`crate::physics::pdn`]) and the per-class
//!   constraint margins (track width vs ampacity, bus skew) already in [`crate::validate`].
//! * **BOM** — bill-of-materials sanity: unique reference designators and a footprint per part.
//!   ([`bom()`])
//!
//! [`verify_all()`] runs every axis and returns a single [`Verification`] whose `all_pass` is the gate.
//!
//! Evidence tier: value-semantic unit tests per axis (a deliberately shorted/open/duplicate board
//! produces the expected finding); the LVS connectivity extraction is differential against the
//! intended netlist. Property/empirical tier, cross-checked by `kicad-cli` in the example flows.
//!
//! # Module layout (Phase 4b carve)
//!
//! The slice is split by verification axis — each sign-off check lives in its own sub-module
//! plus a `suite` aggregator that runs them all and gates the result. The [`pub use`] block at
//! the bottom of this file is the source of truth for the public surface; the table below is the
//! conceptual map.
//!
//! | Sub-module | Axis |
//! |---|---|
//! | `erc` | netlist electrical rule check |
//! | `lvs` | layout-versus-schematic connectivity diff |
//! | `assembly` | courtyard spacing + 3D-model fit + placement-side policy |
//! | `bom` / `keepin` | bill-of-materials sanity + board-edge keep-in |
//! | `decoupling` | PDN decoupling cap proximity to IC power pins |
//! | `isolation` | control ↔ pulser galvanic isolation via BFS |
//! | `ac_coupling` | parasitic coplanar AC coupling to GND return |
//! | `suite` | unified [`Verification`] aggregator running every axis |
//!
//! [`Verification`]: suite::Verification

pub mod ac_coupling;
pub mod assembly;
pub mod bom;
pub mod decoupling;
pub mod erc;
pub mod isolation;
pub mod keepin;
pub mod lvs;
pub mod suite;

#[cfg(test)]
mod tests;

// `pub use` re-exports every public symbol that was previously top-level on the flat
// `src/verify.rs`. Function-name re-exports (e.g. `erc`, `lvs`) collide on the `crate::verify`
// path with the same-named `pub mod` declaration, which makes bare rustdoc links like
// `[`erc`]` ambiguous when written without disambiguating context. The doc-comment links in
// this file use the `[fn]` postfix form (`[`erc()`]`) to bind to the **function** specifically;
// the externally-visible identifier at `crate::verify::erc` is the module, reached via
// `crate::verify::erc::erc()`.
pub use ac_coupling::{parasitic_ac_coupling_check, AcCouplingReport, AcCouplingViolation};
pub use assembly::{assembly, AssemblyReport, AssemblySideViolation, OversizedModel};
pub use bom::{bom, BomReport};
pub use decoupling::{decoupling_proximity, DecouplingReport};
pub use erc::{erc, ErcReport};
pub use isolation::{schematic_isolation_bfs, IsolationReport, IsolationViolation};
pub use keepin::{keepin, KeepinReport};
pub use lvs::{lvs, LvsReport};
pub use suite::{verify_all, Verification};
