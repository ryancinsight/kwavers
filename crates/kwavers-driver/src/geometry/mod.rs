//! Geometry vertical slice (Phase 0 placeholder).
//!
//! # Phase 1+ plan
//!
//! Phase 1 will migrate the flat `src/geom.rs` into this `geometry/` vertical slice, splitting
//! the atomic types (`Nm`, `Mm`, `Point`, `GridSpec`) into their own files and moving
//! distance / convex-hull helpers out of the flat module. The vertical slice brings
//! `src/geometry/` into lock-step with the architectural diagram in `docs/ARCHITECTURE.md`.
//!
//! Phase 0 ships this placeholder so the directory shape is reserved in the source tree.
//! The existing flat `pub mod geom;` declaration in `src/lib.rs` stays authoritative until
//! Phase 1 cuts over (when `src/geom.rs` is renamed to `src/geometry/mod.rs` and this
//! placeholder content replaces it).
//!
//! # SSOT for the slice
//!
//! * `pub mod newtype` — `Nm`, `Mm`, `Point`, `GridSpec` moved out of the current `geom.rs`
//! * `pub mod distance` — `dist_point_seg`, `dist_seg_seg`, `orient`
//! * `pub mod hull` — `convex_hull`
//! * `pub mod tests` — the existing `#[cfg(test)] mod tests` block, with internal types
//!   sourced from `crate::geometry::newtype::*`.
//
// Phase 0 marker — empty placeholder. The actual types live in `src/geom.rs` until the
// Phase 1 cut-over.
