//! Canonical prelude — the `use kwavers_driver::prelude::*;` surface.
//!
//! Phase 1a promotes this from a Phase-0 doc-only file (kept at
//! `docs/PHASE_1_prelude_plan.md`) into a real entry point at `src/prelude.rs`.
//! The goal is one lump-import that brings the canonical unit newtypes, the
//! geometry types, the board model, and the physics facade into scope for
//! downstream examples, integration tests, and the kwavers-backed `experiment` tree.
//!
//! # What lands in scope
//!
//! * **Unit newtypes** — every type-level-unit wrapper from [`crate::units`]:
//!   `Nm` (length), `Hz` (frequency), `Ohm` (impedance), `Watt` (power),
//!   `Kelvin` / `Celsius` (temperature), `Volt`, `Amp`, `Henry`, `Farad`,
//!   `Coulomb`. Together these are the "compile-time units" surface the rest
//!   of the crate rests on.
//! * **Geometry types** — [`Point`], [`GridSpec`] from [`crate::geom`].
//! * **Board model** — [`Board`], [`LayerId`], [`NetId`], [`NetClassKind`],
//!   [`Pad`], [`Track`], [`Via`], [`ViaKind`] from [`crate::board`].
//! * **Physics facade** — the per-domain physics kernels stay in their
//!   modules; the prelude does **not** glob them out (so a downstream
//!   `use kwavers_driver::*` still imports everything, but the prelude stays
//!   narrow + focused on stable entry points).
//!
//! # What is deliberately left out
//!
//! * Every existing physics module's helper function (e.g.
//!   `pulser_dissipation`, `microstrip_impedance`, `ir_drop`, …). Those are
//!   still reached through the crate-root [`crate`] — today's contract is
//!   "drop into any module's namespace" rather than "glob the whole crate".
//! * The crate-side `error::Error` / `error::Result` — those land in the
//!   prelude in Phase 1b alongside the per-vertical-slice error hierarchy.
//! * The `experiment` tree top-level — that lands at Phase 5.
//!
//! # SSOT marker
//!
//! If a downstream consumer only knows *one* kwavers-driver import path,
//! this is it: `use kwavers_driver::prelude::*;` lands every public type and
//! surface that Phase 1a commits to.

// ── Unit newtypes (the compile-time-units surface) ────────────────────────
pub use crate::units::{Amp, Celsius, Coulomb, Farad, Henry, Hz, Kelvin, Nm, Ohm, Volt, Watt};

// ── Geometry (length-aware board coordinate types) ────────────────────────
pub use crate::geom::{GridSpec, Point};

// ── Board model (the canonical routing domain) ────────────────────────────
pub use crate::board::{
    Board, LayerId, Net, NetClassKind, NetId, Pad, SplitDomain, Track, Via, ViaKind,
};
