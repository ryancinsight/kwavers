//! The [`RoutingCost`] trait — the cost-model dependency-inversion seam this router depends on.
//!
//! See the [module-level documentation](super) for the full cost-seam rationale + the
//! proximity-hazard-model summary.
//!
//! [`RoutingCost`] is the extension point that makes this router *physics-guided*: the
//! negotiated-congestion search multiplies a node's congestion penalty by its **base cost**, and
//! the base cost is where design intent lives. The router depends on this trait, *never* on a
//! concrete cost — concrete implementations (e.g. [`PhysicsCost`](super::physics::PhysicsCost))
//! live alongside it, and a future thermal-aware or impedance-aware cost is a new implementor,
//! not a change to the router.

use crate::board::{LayerId, NetClassKind};
use crate::geom::Point;

/// Cost model consumed by the router. Implementors provide the intrinsic (congestion-independent)
/// cost of occupying a node, plus the cost of inserting a via.
///
/// This is a deliberate role interface (DIP): the router depends on this trait, never on a
/// concrete cost. A future thermal-aware or impedance-aware cost is a new implementor, not a
/// change to the router.
pub trait RoutingCost {
    /// Intrinsic cost (`>= 0`) of routing net class `class` through board point `p` on `layer`,
    /// independent of present congestion. Returns [`f64::INFINITY`] for a hard keepout.
    fn node_base(&self, p: Point, layer: LayerId, class: NetClassKind) -> f64;

    /// Cost charged for a layer transition (via) on `class`.
    ///
    /// The default is class-neutral. Physics-aware implementations can charge high-speed classes
    /// more heavily because vias introduce impedance discontinuities and must be minimized.
    fn via_cost(&self, _class: NetClassKind) -> f64 {
        10.0
    }
}
