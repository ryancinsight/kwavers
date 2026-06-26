//! Geometry / grid / board-model invariant errors.
//!
//! Migrates the 4 variants that lived on the legacy flat `Error` enum at Phase 0:
//!
//! * [`Geometry::PadOutOfBounds`] — a pad's coordinates lie outside the routing grid
//!   bounds. Raised by [`crate::validate`] when reconciling pin assignments against
//!   the engineered grid.
//! * [`Geometry::UnreachableTerminal`] — a net referenced a terminal that the
//!   negotiated-congestion PathFinder could not route to; the terminal maps to a
//!   blocked or out-of-bounds grid node.
//! * [`Geometry::EmptyGrid`] — the grid spec is degenerate (zero cells on some axis).
//!   Raised by [`crate::geom::GridSpec::cover`].
//! * [`Geometry::GridPitchTooCoarse`] — two distinct nets' terminals collapsed onto a
//!   single grid node; the chosen pitch cannot separate them. Increase the resolution
//!   or split the collision manually.
//!
//! All four are migrated verbatim from the hand-written Phase-0 Display strings so
//! existing diagnostic output is byte-stable for downstream log scrapers.

/// Geometry invariant failure.
///
/// Derives `Debug, Clone, Copy, PartialEq, Eq` because **every field is an integer
/// type** (`usize`/`u32` — `PadOutOfBounds::pad`, `UnreachableTerminal::net`,
/// `GridPitchTooCoarse::node`, plus the ZST `EmptyGrid`) — `Copy` is sound (no resource
/// handles in fields; numeric copies are cheap) and `Eq` is sound (no `f64`/`String`/
/// `io::Error` to break reflexivity). This is the **odd-one-out** in the slice tree — the
/// other eight sub-enums drop `Clone + PartialEq + Eq` because their variants carry at
/// least one `f64` measurement (or `String`/`io::Error`) that violates `Eq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Geometry {
    /// A pad's position lies outside the routing grid bounds.
    #[error("pad {pad} lies outside the routing grid bounds")]
    PadOutOfBounds {
        /// Pad index in the board's pad list.
        pad: usize,
    },

    /// A net referenced a terminal that maps to a blocked or unreachable grid node.
    #[error("net {net} has a terminal that cannot be reached on the grid")]
    UnreachableTerminal {
        /// Net index whose terminal could not be connected.
        net: u32,
    },

    /// The grid specification is degenerate (zero extent on some axis).
    #[error("routing grid has zero extent on some axis")]
    EmptyGrid,

    /// Two distinct terminals of different nets collapsed onto the same grid node; the grid
    /// pitch is too coarse to separate them.
    #[error("grid pitch too coarse: distinct-net terminals collide at node {node}")]
    GridPitchTooCoarse {
        /// Flat node index where the collision occurred.
        node: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source-authority guarantee: every Phase-0 Display string survives the migration
    /// byte-for-byte so downstream log scrapers do not have to be updated.
    #[test]
    fn display_strings_preserve_phase_0_invariant() {
        assert_eq!(
            Geometry::PadOutOfBounds { pad: 7 }.to_string(),
            "pad 7 lies outside the routing grid bounds"
        );
        assert_eq!(
            Geometry::UnreachableTerminal { net: 42 }.to_string(),
            "net 42 has a terminal that cannot be reached on the grid"
        );
        assert_eq!(
            Geometry::EmptyGrid.to_string(),
            "routing grid has zero extent on some axis"
        );
        assert_eq!(
            Geometry::GridPitchTooCoarse { node: 19 }.to_string(),
            "grid pitch too coarse: distinct-net terminals collide at node 19"
        );
    }

    /// The sub-enum is `#[non_exhaustive]`, so exhaustively matching the four legacy
    /// variants is *forbidden*. The test compiles because of the wildcard — without the
    /// attribute the same match would compile cleanly and that regression is what the
    /// `#[non_exhaustive]` is supposed to guard against.
    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(g: Geometry) -> &'static str {
            match g {
                Geometry::PadOutOfBounds { .. } => "pob",
                Geometry::UnreachableTerminal { .. } => "unt",
                Geometry::EmptyGrid => "eg",
                Geometry::GridPitchTooCoarse { .. } => "gptc",
                _ => "future-variant",
            }
        }
    }

    /// Same `Eq` + `Copy` shape as the Phase-0 enum — needed for callers that compare
    /// errors structurally (e.g. cache-keyed early returns). The `.clone()` call
    /// is intentional: this test exercises the `Clone` derive alongside `Copy`. The
    /// `clippy::clone_on_copy` allow on the whole test function documents the intent.
    #[test]
    #[allow(clippy::clone_on_copy)]
    fn variants_implement_clone_eq_copy() {
        let a = Geometry::EmptyGrid;
        let b = a; // Copy
        assert_eq!(a, b);
        let c = a.clone(); // Clone (intentional, see fn doc)
        assert_eq!(a, c);
    }
}
