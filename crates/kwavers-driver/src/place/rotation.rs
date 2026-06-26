//! Placement rotation primitives: the 90°-step orientation enum + the footprint's
//! rotation-freedom policy.
//!
//! Phase 2c carved this out of `src/place/footprint.rs` per the spec's
//! `place/{mod, anneal, energy, footprint, import, rotation, tests}.rs` layout.
//! The `Rot` enum is a 4-variant ZST marker (zero bytes; the four 90° orientation
//! steps for KiCad footprint-angle emit). The `RotationPolicy` enum is a 3-variant
//! placement-rotation-freedom marker (Fixed / HalfTurn / AnyRightAngle) plus the
//! role-default helper [`RotationPolicy::for_role`].
//!
//! Both stay `pub` so the `crate::place::{Rot, RotationPolicy}` re-export surface at
//! `src/place/mod.rs` is unchanged — `pub use rotation::{Rot, RotationPolicy}` is the
//! post-Phase-2c routing, and the pre-Phase-2c form
//! `pub use footprint::{Rot, RotationPolicy}` simply moves one folder down.
//!
//! (Plain backticks throughout for crate-internal references — same convention as
//! the cost slice + the route slice. The FootprintDef struct itself lives at
//! [`crate::place::footprint::FootprintDef`].)

// `LayerId` is unused here — `Rot` itself has no data, and the `degrees()`
// method returns `f64` for the KiCad footprint-angle emit (no per-pad layer state).

/// A 90° orientation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Rot {
    /// 0°.
    #[default]
    R0,
    /// 90° counter-clockwise.
    R90,
    /// 180°.
    R180,
    /// 270° counter-clockwise.
    R270,
}

impl Rot {
    /// Orientation in degrees (CCW), for emitting a KiCad footprint angle.
    #[must_use]
    pub fn degrees(self) -> f64 {
        match self {
            Rot::R0 => 0.0,
            Rot::R90 => 90.0,
            Rot::R180 => 180.0,
            Rot::R270 => 270.0,
        }
    }

    /// The next 90° step (cycles R0→R90→R180→R270→R0).
    #[must_use]
    pub fn next(self) -> Rot {
        match self {
            Rot::R0 => Rot::R90,
            Rot::R90 => Rot::R180,
            Rot::R180 => Rot::R270,
            Rot::R270 => Rot::R0,
        }
    }

    /// The 180° counterpart.
    #[must_use]
    pub fn opposite(self) -> Rot {
        match self {
            Rot::R0 => Rot::R180,
            Rot::R90 => Rot::R270,
            Rot::R180 => Rot::R0,
            Rot::R270 => Rot::R90,
        }
    }

    /// Next rotation allowed by a footprint policy, relative to the floorplanned orientation.
    #[must_use]
    pub fn next_allowed(self, initial: Rot, policy: RotationPolicy) -> Option<Rot> {
        match policy {
            RotationPolicy::Fixed => None,
            RotationPolicy::HalfTurn => Some(if self == initial {
                initial.opposite()
            } else {
                initial
            }),
            RotationPolicy::AnyRightAngle => Some(self.next()),
        }
    }

    /// Rotate an offset about the footprint centre.
    #[must_use]
    pub fn apply(self, p: crate::geom::Point) -> crate::geom::Point {
        match self {
            Rot::R0 => p,
            Rot::R90 => crate::geom::Point::new(-p.y, p.x),
            Rot::R180 => crate::geom::Point::new(-p.x, -p.y),
            Rot::R270 => crate::geom::Point::new(p.y, -p.x),
        }
    }

    /// Apply to a `(width, height)` courtyard size — swaps axes for the quarter turns.
    #[must_use]
    pub fn apply_size(
        self,
        size: (crate::geom::Nm, crate::geom::Nm),
    ) -> (crate::geom::Nm, crate::geom::Nm) {
        match self {
            Rot::R0 | Rot::R180 => size,
            Rot::R90 | Rot::R270 => (size.1, size.0),
        }
    }
}

/// Rotation freedom a footprint may use during placement optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationPolicy {
    /// Preserve the schematic/floorplan orientation exactly.
    Fixed,
    /// Permit only the original orientation and its 180° counterpart.
    HalfTurn,
    /// Permit all four right-angle orientations.
    AnyRightAngle,
}

impl RotationPolicy {
    /// Default policy for the placement role.
    #[must_use]
    pub fn for_role(role: super::footprint::Role) -> Self {
        match role {
            super::footprint::Role::ActiveIc
            | super::footprint::Role::Connector
            | super::footprint::Role::Power => RotationPolicy::Fixed,
            super::footprint::Role::Decoupling | super::footprint::Role::Passive => {
                RotationPolicy::HalfTurn
            }
        }
    }
}
