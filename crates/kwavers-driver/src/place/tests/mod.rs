//! Slice-wide tests for the `place` slice.
//!
//! Phase 2c consolidated the previously-inline `mod tests { ... }` blocks of
//! `src/place/{mod, footprint, footprint_import, component, symbol_import}.rs`
//! into a single `crate::place::tests` module. The pattern matches the Phase 2a
//! `cost::tests` / Phase 2b `route::tests` migration: one slice-wide test file
//! collects the topology tests + parser pinning tests + integration tests so an
//! external reviewer can grep `tests.rs` once and see the entire place-slice
//! behaviour contract.
//!
//! The S-expression kernel (`Sexpr`, `parse_sexpr`, `child`, `num`, `xyz_child`) now lives at
//! `crate::place::sexpr` (`pub(crate)`) — the SSOT shared with `io::pcb_parse`. The byte-tracking
//! pinning tests here (`parse_sexpr_unclosed_*`,
//! `parse_sexpr_unicode_byte_offset_differs_from_char_offset`,
//! `imported_model_offset_is_recentered_with_pads`,
//! `imports_model_offset_and_rotation`) reach them via `use crate::place::sexpr::*`.

use crate::board::{LayerId, NetId};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::sexpr::{child, parse_sexpr, xyz_child};
use crate::place::{
    anneal, energy as energy_fn, import_kicad_mod, import_symbol_pinmap, AnnealParams, Component,
    CongestionField, FootprintDef, PadDef, PinMap, PlaceConfig, PlaceWeights, Placement, Rect,
    Role, Rot, RotationPolicy,
};

mod energy;
mod geometry;
mod import;
mod rotation;

// ─── Fixtures (lifted from inline `mod tests` blocks of mod.rs / footprint.rs / component.rs) ───

pub(super) fn ic(name: &str) -> FootprintDef {
    FootprintDef::new(
        name,
        (Nm::from_mm(8.0), Nm::from_mm(8.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(4.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.6), Nm::from_mm(0.6)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    )
}

pub(super) fn conn(name: &str) -> FootprintDef {
    FootprintDef::new(
        name,
        (Nm::from_mm(5.0), Nm::from_mm(16.0)),
        Role::Connector,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(0.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(1.0), Nm::from_mm(1.0)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )
}

pub(super) fn comp(fp: usize, refdes: &str, x: f64, y: f64) -> Component {
    Component {
        fp,
        nets: vec![None],
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }
}

/// Distance (mm) from a component's courtyard to the nearest board edge.
pub(super) fn to_edge(c: &Component, lib: &[FootprintDef], board: (Nm, Nm)) -> f64 {
    let r = c.courtyard(lib);
    let w = board.0.to_mm();
    let h = board.1.to_mm();
    r.min
        .x
        .to_mm()
        .min(w - r.max.x.to_mm())
        .min(r.min.y.to_mm())
        .min(h - r.max.y.to_mm())
}

/// One pad-row of an N-pad footprint at `pitch`, used to exercise the fine-pitch escape predicate.
pub(super) fn row_fp(pitch_mm: f64, n: usize) -> FootprintDef {
    let pads = (0..n)
        .map(|k| PadDef {
            offset: Point::new(Nm::from_mm(k as f64 * pitch_mm), Nm(0)),
            size: (Nm::from_mm(0.3), Nm::from_mm(0.3)),
            layers: vec![LayerId(0)],
            power_pin: false,
        })
        .collect();
    FootprintDef::new(
        "row",
        (Nm::from_mm(n as f64 * pitch_mm), Nm::from_mm(1.0)),
        Role::ActiveIc,
        pads,
    )
}

pub(super) fn lib() -> Vec<FootprintDef> {
    vec![FootprintDef::new(
        "U",
        (Nm::from_mm(8.0), Nm::from_mm(4.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm::from_mm(3.0), Nm::from_mm(0.0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: true,
        }],
    )]
}
