//! Placed component instances and their geometry.

use crate::board::{LayerId, NetId};
use crate::geom::{Nm, Point};
use crate::place::footprint::{FootprintDef, IsolationDomain};
use crate::place::Rot;

/// Whether a reference designator denotes a diode/TVS surge suppressor.
#[must_use]
pub(crate) fn is_surge_suppressor_refdes(refdes: &str) -> bool {
    refdes.starts_with('D') || refdes.starts_with("TVS")
}

/// Whether a reference designator denotes a crystal/resonator/oscillator support component.
#[must_use]
pub(crate) fn is_crystal_refdes(refdes: &str) -> bool {
    refdes.starts_with('X') || refdes.starts_with('Y')
}

/// A placement: where a component sits and how it is turned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Placement {
    /// Footprint-centre position on the board.
    pub pos: Point,
    /// Orientation.
    pub rot: Rot,
}

/// A component instance: a footprint (by library index), per-pad net assignment, a placement, and
/// an optional association to the IC it decouples.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Component {
    /// Index into the footprint library.
    pub fp: usize,
    /// Per-pad nets, aligned to the footprint's `pads` (`None` = unconnected).
    pub nets: Vec<Option<NetId>>,
    /// Reference designator (e.g. `"U1"`), for emit/debug.
    pub refdes: String,
    /// Current placement.
    pub placement: Placement,
    /// For a decoupling cap: the component index of the IC it bypasses.
    pub assoc_ic: Option<usize>,
    /// Fixed-position: the placer must not move or rotate this instance. Used for inter-tile mating
    /// connectors that must sit at an identical board position on every stacked tile so they mate.
    pub locked: bool,
    /// LV↔HV isolation domain — drives the placement `t.isolation_drift` term so LV components
    /// stay on the axis-min side of [`crate::place::energy::PlaceConfig::isolation_axis`] and HV on
    /// the axis-max side. Default [`IsolationDomain::Lv`] (preserved by struct-update sites that
    /// use `..Component::default()`); opt in for HV parts via [`Self::with_isolation_domain`].
    pub isolation_domain: IsolationDomain,
}

/// An axis-aligned rectangle `[min, max]` in board coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Lower-left corner.
    pub min: Point,
    /// Upper-right corner.
    pub max: Point,
}

impl Rect {
    /// Grow the rectangle by `d` on every side (a clearance ring).
    #[must_use]
    pub fn inflate(self, d: Nm) -> Rect {
        Rect {
            min: Point::new(self.min.x - d, self.min.y - d),
            max: Point::new(self.max.x + d, self.max.y + d),
        }
    }

    /// Overlap area with another rectangle, in nm² (`0` if disjoint).
    #[must_use]
    pub fn overlap_area(self, other: Rect) -> f64 {
        let ox = (self.max.x.0.min(other.max.x.0) - self.min.x.0.max(other.min.x.0)).max(0);
        let oy = (self.max.y.0.min(other.max.y.0) - self.min.y.0.max(other.min.y.0)).max(0);
        ox as f64 * oy as f64
    }
}

/// Component-to-component assembly clearance violation.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentClearanceViolation {
    /// First reference designator.
    pub first: String,
    /// Second reference designator.
    pub second: String,
    /// Overlap area of the clearance-inflated courtyards, in square millimetres.
    pub overlap_mm2: f64,
}

/// Find component pairs whose courtyards violate the required assembly clearance.
///
/// Each courtyard is inflated by half the required clearance; a non-zero overlap then means the
/// original courtyards are closer than `clearance`, including the hard intersection case.
#[must_use]
pub fn component_clearance_violations(
    comps: &[Component],
    lib: &[FootprintDef],
    clearance: Nm,
) -> Vec<ComponentClearanceViolation> {
    let half = Nm(clearance.0 / 2);
    let mut violations = Vec::new();
    for (i, a) in comps.iter().enumerate() {
        let ar = a.courtyard(lib).inflate(half);
        for b in comps.iter().skip(i + 1) {
            let overlap_mm2 = ar.overlap_area(b.courtyard(lib).inflate(half)) * 1.0e-12;
            if overlap_mm2 > 0.0 {
                violations.push(ComponentClearanceViolation {
                    first: a.refdes.clone(),
                    second: b.refdes.clone(),
                    overlap_mm2,
                });
            }
        }
    }
    violations
}

impl Component {
    /// Absolute position of pad `k` under the current placement.
    #[must_use]
    pub fn pad_pos(&self, lib: &[FootprintDef], k: usize) -> Point {
        let off = self.placement.rot.apply(lib[self.fp].pads[k].offset);
        Point::new(self.placement.pos.x + off.x, self.placement.pos.y + off.y)
    }

    /// Iterate `(absolute position, layers, net)` for every pad.
    pub fn placed_pads<'a>(
        &'a self,
        lib: &'a [FootprintDef],
    ) -> impl Iterator<Item = (Point, &'a [LayerId], Option<NetId>)> + 'a {
        let fp = &lib[self.fp];
        fp.pads
            .iter()
            .enumerate()
            .map(move |(k, pad)| (self.pad_pos(lib, k), pad.layers.as_slice(), self.nets[k]))
    }

    /// Courtyard rectangle under the current placement.
    #[must_use]
    pub fn courtyard(&self, lib: &[FootprintDef]) -> Rect {
        let (w, h) = self.placement.rot.apply_size(lib[self.fp].courtyard);
        let hw = Nm(w.0 / 2);
        let hh = Nm(h.0 / 2);
        Rect {
            min: Point::new(self.placement.pos.x - hw, self.placement.pos.y - hh),
            max: Point::new(self.placement.pos.x + hw, self.placement.pos.y + hh),
        }
    }

    /// Builder: tag this component as an HV-side part. Used by examples and tests to opt into
    /// the isolation-barrier placer (`t.isolation_drift`). Default is LV; an HV tag park the
    /// component on the axis-max side of [`crate::place::energy::PlaceConfig::isolation_axis`].
    #[must_use]
    pub fn with_isolation_domain(mut self, domain: IsolationDomain) -> Self {
        self.isolation_domain = domain;
        self
    }
}
