//! Canonical prelude — the `use kwavers_driver::prelude::*;` surface.
//!
//! One import that brings the canonical unit newtypes, the geometry types, and the board
//! model into scope for downstream examples, integration tests, and the kwavers-backed
//! `experiment` tree.
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
//! * The crate-side `error::Error` / `error::Result`, reached through [`crate::error`].
//! * The `experiment` tree top level, reached through [`crate::experiment`].
//!
//! # SSOT marker
//!
//! If a downstream consumer knows only *one* kwavers-driver import path, this is it:
//! `use kwavers_driver::prelude::*;` lands the unit, geometry, and board types listed
//! above.

// ── Unit newtypes (the compile-time-units surface) ────────────────────────
pub use crate::units::{Amp, Celsius, Coulomb, Farad, Henry, Hz, Kelvin, Nm, Ohm, Volt, Watt};

// ── Geometry (length-aware board coordinate types) ────────────────────────
pub use crate::geom::{GridSpec, Point};

// ── Board model (the canonical routing domain) ────────────────────────────
pub use crate::board::{
    Board, LayerId, Net, NetClassKind, NetId, Pad, SplitDomain, Track, Via, ViaKind,
};
