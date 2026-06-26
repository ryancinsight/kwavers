//! `io` slice facade — KiCad file emission.
//!
//! Phase 4a carve: `src/io.rs` was split into a 7-file `src/io/` subtree by **emission
//! format** (pcb / sch / dru / pro / project), with this `mod.rs` carrying the format-agnostic
//! helpers that every emission kernel reaches via the `#[macro_use]` scope-inheritance trick:
//!
//! * `MechKind` / `MechFeature` + `mechanical_features` — the SSOT for board mechanical
//!   features (4 corner MountingHole + 3 fiducials) shared by [`crate::io::pcb_emit`] (PCB
//!   emission) AND by [`crate::pipeline`] (router keepout reservation).
//! * `Uuid` — sequential, deterministic UUID generator (no clock / RNG; keeps emission
//!   reproducible across runs).
//! * `w!` / `wln!` macros — the `String`-buffer `format!` writer helpers used by every
//!   emission submodule so the writers carry no `fmt::Result` plumbing on each line.
//! * `layer_name` / `layer_ordinal` / `mm` — KiCad layer-ordinal translation (F.Cu=0,
//!   In{n}=n, B.Cu=31) + the `nm → mm` factor.
//! * `duplicate_pcb_uuids` — the post-emission sanity check (KiCad's footprints/pads/tracks
//!   identify by UUID and re-using the same UUID across distinct objects corrupts the
//!   ratsnest and DRC diagnostics).
//!
//! # Feature gate
//!
//! The whole slice is gated behind the `io` Cargo feature. The default build enables `io`
//! (so today's call sites continue to compile unchanged); `--no-default-features` skips the
//! slice fat. Off-build requires further gating of the four caller sites in
//! [`crate::pipeline`] + `crate::place::energy` — tracked in `docs/MIGRATION.md` as a
//! follow-up.
//!
//! # Per-format sub-module split
//!
//! | Sub-module | Public API |
//! |---|---|
//! | `pcb_emit` | `write_kicad_pcb` + `save_kicad_pcb` + `save_kicad_pcb_flagged` + the `emit_footprint_body` helper |
//! | `sch_emit` | `write_kicad_sch` + `save_kicad_sch` |
//! | `pro_emit` | `write_kicad_pro` |
//! | `dru_emit` | `write_kicad_dru` |
//! | `project_emit` | `save_kicad_project` + `save_kicad_project_flagged` |
//!
//! # Format version stamping
//!
//! The KiCad format-version literals (`KICAD_PCB_FORMAT_VERSION`, `KICAD_SCH_FORMAT_VERSION`)
//! and the generator name (`KICAD_GENERATOR_NAME`) moved to [`crate::ssot`] at Phase 1c. A future
//! KiCad version bump touches one SSOT address, not every emission submodule.

// ============================================================================
// Macro definitions at the TOP of the file (so every emission submodule sees them
// via the `#[macro_use] pub mod …;` scope-inheritance trick declared at the bottom).
// ============================================================================

/// `String`-buffer `writeln!` helper: `String`'s `write_fmt` is infallible, so this collapses
/// the `Result` plumbing at every emission call site. Inherited by every emission submodule
/// via the `#[macro_use]` declaration at the bottom of this file.
macro_rules! wln {
    ($dst:expr, $($arg:tt)*) => {
        writeln!($dst, $($arg)*).expect("invariant: String write_fmt never fails")
    };
}

/// Single-line variant of [`wln!`] — calls `write!` instead of `writeln!`. Same infallibility
/// rationale.
macro_rules! w {
    ($dst:expr, $($arg:tt)*) => {
        write!($dst, $($arg)*).expect("invariant: String write_fmt never fails")
    };
}

// ============================================================================
// Format-agnostic helper items at the slice facade.
//
// KiCad format-version stamps (`KICAD_PCB_FORMAT_VERSION`, `KICAD_SCH_FORMAT_VERSION`) and
// the generator name (`KICAD_GENERATOR_NAME`) live at `crate::ssot::*` (Phase 1c SSOT). Each
// emission submodule declares its OWN `use crate::ssot::*;` for the format-string
// interpolations that reference it (a `use` does NOT carry over `#[macro_use]` propagation).
// No slice-facade-level `use crate::ssot::*;` is needed here.
// ============================================================================

/// Sequential, deterministic UUIDs (no clock / RNG — keeps emission reproducible).
pub(super) struct Uuid(pub(super) u64);
impl Uuid {
    pub(super) fn next(&mut self) -> String {
        self.0 += 1;
        format!("00000000-0000-0000-0000-{:012x}", self.0)
    }
}

/// KiCad copper-layer name for `LayerId(l)` on an `nlayers` stack: layer 0 = `F.Cu`, the last =
/// `B.Cu`, inner layers = `In{l}.Cu`. Pub(super) so `pcb_emit` + `sch_emit` reach it.
pub(super) fn layer_name(l: u16, nlayers: usize) -> String {
    if l == 0 {
        "F.Cu".to_string()
    } else if l as usize == nlayers - 1 {
        "B.Cu".to_string()
    } else {
        format!("In{l}.Cu")
    }
}

/// KiCad layer ordinal (v7/8 numbering): F.Cu=0, In{n}=n, B.Cu=31.
pub(super) fn layer_ordinal(l: u16, nlayers: usize) -> u32 {
    if l as usize == nlayers - 1 {
        31
    } else {
        l as u32
    }
}

/// Nanometre → millimetre factor (the engine's `Nm` storage is integer nm).
#[inline]
pub(super) fn mm(p: i64) -> f64 {
    p as f64 * 1.0e-6
}

/// Kind of board mechanical feature (drives its emitted footprint and copper keepout radius).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechKind {
    /// Pick-and-place vision target (small SMD copper dot, soldermask opening).
    Fiducial,
    /// Non-plated mounting hole (3.2 mm drill) for screw fixing.
    MountingHole,
}

/// A board mechanical feature at `(x, y)` mm.
#[derive(Debug, Clone, Copy)]
pub struct MechFeature {
    /// X position (mm).
    pub x: f64,
    /// Y position (mm).
    pub y: f64,
    /// Feature kind.
    pub kind: MechKind,
}

impl MechFeature {
    /// Copper-keepout radius (mm) the router must hold clear around the feature: the fiducial pad
    /// plus clearance, or the mounting-hole barrel plus a washer-class annulus.
    #[must_use]
    pub fn keepout_mm(&self) -> f64 {
        match self.kind {
            MechKind::Fiducial => 1.5,
            MechKind::MountingHole => 2.8,
        }
    }
}

/// The board's mechanical features (3 fiducials in an L for unambiguous orientation + 4 corner
/// mounting holes), positioned relative to the board size `(w_mm, h_mm)` so they scale with the
/// board. Single source of truth shared by [`crate::io::pcb_emit::write_kicad_pcb`] (emission)
/// and the router (keepout reservation in [`crate::pipeline`]/`crate::place::energy`), so the
/// holes the router avoids are exactly the holes that get drilled.
#[must_use]
pub fn mechanical_features(w_mm: f64, h_mm: f64) -> Vec<MechFeature> {
    // Inset corners proportionally but clamped, so the features stay near the corners on a large
    // board yet remain inside a small one.
    let mi = (w_mm.min(h_mm) * 0.09).clamp(3.0, 8.0); // mounting-hole inset
    let fi = mi + 3.0; // fiducials sit just inboard of the mounting holes
    let f = |x, y, kind| MechFeature { x, y, kind };
    vec![
        f(mi, mi, MechKind::MountingHole),
        f(w_mm - mi, mi, MechKind::MountingHole),
        f(mi, h_mm - mi, MechKind::MountingHole),
        f(w_mm - mi, h_mm - mi, MechKind::MountingHole),
        f(fi, fi, MechKind::Fiducial),
        f(w_mm - fi, fi, MechKind::Fiducial),
        f(fi, h_mm - fi, MechKind::Fiducial),
    ]
}

/// Return duplicate UUID values from a KiCad PCB text, sorted lexicographically.
///
/// KiCad identifies board objects by UUID. Reusing a UUID across footprints, pads, tracks, or vias can
/// make the ratsnest and DRC diagnostics refer to the wrong object, so generated boards are not
/// trustworthy until this returns an empty vector.
#[must_use]
pub fn duplicate_pcb_uuids(pcb_text: &str) -> Vec<String> {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let needle = "(uuid";
    let mut rest = pcb_text;
    while let Some(idx) = rest.find(needle) {
        rest = &rest[idx + needle.len()..];
        let trimmed = rest.trim_start();
        let Some(after_quote) = trimmed.strip_prefix('"') else {
            rest = trimmed;
            continue;
        };
        let mut escaped = false;
        let mut end = None;
        for (i, c) in after_quote.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            if c == '\\' {
                escaped = true;
                continue;
            }
            if c == '"' {
                end = Some(i);
                break;
            }
        }
        let Some(end) = end else {
            break;
        };
        let value = &after_quote[..end];
        if is_uuid_shape(value) {
            *counts.entry(value.to_string()).or_default() += 1;
        }
        rest = &after_quote[end + 1..];
    }

    counts
        .into_iter()
        .filter_map(|(uuid, count)| (count > 1).then_some(uuid))
        .collect()
}

fn is_uuid_shape(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    bytes.iter().enumerate().all(|(i, b)| match i {
        8 | 13 | 18 | 23 => *b == b'-',
        _ => b.is_ascii_hexdigit(),
    })
}// Facade-level tests live in `src/io/tests.rs` (the single slice-wide test surface).
// Dropping per-`mod.rs` fixtures eliminates the previously-duplicated coverage
// (`duplicate_pcb_uuids_reports_only_repeated_values` ↔
//  `tests.rs::duplicate_pcb_uuids_reports_only_repeated_uuid_values`).
#[cfg(test)]
mod facade_tests {}

// ============================================================================
// Sub-module declarations at the BOTTOM of the file (idiomatic Rust ordering).
//
// Each `#[macro_use] pub mod …;` brings the `w!`/`wln!` macros declared at the top
// of this file into the submodule's lexical scope via Rust 2018+ scope-inheritance.
// ============================================================================

#[macro_use]
pub mod dru_emit;
#[macro_use]
pub mod pcb_emit;
#[macro_use]
pub mod pro_emit;
#[macro_use]
pub mod project_emit;
#[macro_use]
pub mod sch_emit;

// Re-export the per-format kernels so callers can reach them at `crate::io::*`
// without traversing the per-format submodules directly. `save_kicad_project`
// is the orchestrator (saves pcb + dru + pro sidecars together) and lives in
// `project_emit`; the rest sit alongside their emission format.
pub use dru_emit::write_kicad_dru;
pub use pcb_emit::{save_kicad_pcb, save_kicad_pcb_flagged, write_kicad_pcb};
pub use pro_emit::write_kicad_pro;
pub use project_emit::{save_kicad_project, save_kicad_project_flagged};
pub use sch_emit::{save_kicad_sch, write_kicad_sch};
