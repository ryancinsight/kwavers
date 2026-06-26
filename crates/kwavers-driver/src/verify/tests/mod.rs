//! Slice-wide tests for the [`verify`](super) facade.
//!
//! Shared helpers (`spec`, `two_pad_fp`, `comp`, `comp_at`) live here; per-axis test families
//! are split into sub-modules: `assembly` (physical verification) and `electrical` (ERC/BOM/
//! schematic-isolation/AC-coupling). `lvs` was split in an earlier session.
use super::*;
use crate::board::{LayerId, NetId};
use crate::geom::{GridSpec, Nm, Point};
use crate::place::component::{Component, Placement};
use crate::place::footprint::{FootprintDef, PadDef, Role};
use crate::place::rotation::Rot;

mod assembly;
mod electrical;
mod lvs;

pub(super) fn spec() -> GridSpec {
    GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap()
}

pub(super) fn two_pad_fp() -> FootprintDef {
    FootprintDef::new(
        "R",
        (Nm::from_mm(2.0), Nm::from_mm(1.0)),
        Role::Passive,
        vec![
            PadDef {
                offset: Point::new(Nm::from_mm(-0.5), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
            PadDef {
                offset: Point::new(Nm::from_mm(0.5), Nm(0)),
                size: (Nm::from_mm(0.4), Nm::from_mm(0.4)),
                layers: vec![LayerId(0)],
                power_pin: false,
            },
        ],
    )
    .with_model("m.step", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
}

pub(super) fn comp(fp: usize, refdes: &str, nets: Vec<Option<NetId>>) -> Component {
    Component {
        fp,
        nets,
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(10.0), Nm::from_mm(10.0)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    }
}

pub(super) fn comp_at(fp: usize, refdes: &str, x: f64, y: f64, assoc: Option<usize>) -> Component {
    Component {
        fp,
        nets: Vec::new(),
        refdes: refdes.into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: assoc,
        locked: false,
        ..Default::default()
    }
}
