//! Commutation-loop scene walker — [`CommutationLoop`] struct + [`commutation_loops`] walker
//! + the placement-aware `pad_on_net` helper.

use crate::board::NetId;
use crate::geom::Point;
use crate::place::{
    component::Component,
    footprint::{FootprintDef, Role},
};
use super::r#loop::{loop_inductance_nh, polygon_area_mm2};

/// A device's commutation loop: its enclosed area and a representative location.
#[derive(Debug, Clone, Copy)]
pub struct CommutationLoop {
    /// Loop area (mm²).
    pub area_mm2: f64,
    /// First-order loop-inductance estimate (nH).
    pub inductance_nh: f64,
    /// Where the loop sits (the cap centre) — for hotspot feedback.
    pub at: Point,
}

/// First pad of `comp` on net `net`, if any. Slice-private scene-walker helper.
fn pad_on_net(comp: &Component, lib: &[FootprintDef], net: NetId) -> Option<Point> {
    comp.placed_pads(lib)
        .find(|(_, _, n)| *n == Some(net))
        .map(|(p, _, _)| p)
}

/// Commutation loops for every decoupling cap tied to a device.
#[must_use]
pub fn commutation_loops(comps: &[Component], lib: &[FootprintDef]) -> Vec<CommutationLoop> {
    let mut loops = Vec::new();
    for cap in comps {
        if !matches!(lib[cap.fp].role, Role::Decoupling) {
            continue;
        }
        let Some(ic_idx) = cap.assoc_ic else { continue };
        // The cap's two terminals (and their nets) are the loop's near side.
        let nets: Vec<NetId> = cap.nets.iter().flatten().copied().collect();
        if nets.len() != 2 {
            continue;
        }
        let ic = &comps[ic_idx];
        let (Some(cap_a), Some(cap_b)) =
            (pad_on_net(cap, lib, nets[0]), pad_on_net(cap, lib, nets[1]))
        else {
            continue;
        };
        let (Some(ic_a), Some(ic_b)) = (pad_on_net(ic, lib, nets[0]), pad_on_net(ic, lib, nets[1]))
        else {
            continue;
        };
        // Loop: cap_a → ic_a → ic_b → cap_b.
        let area = polygon_area_mm2(&[cap_a, ic_a, ic_b, cap_b]);
        loops.push(CommutationLoop {
            area_mm2: area,
            inductance_nh: loop_inductance_nh(area),
            at: cap.placement.pos,
        });
    }
    loops
}
