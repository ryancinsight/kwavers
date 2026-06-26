//! Per-tile transducer dispatch and lane binding (Phase 5).
//!
//! Maps `manifest.tx_nets[lane]` → tile index for a balanced equal-partition stack.
//! For the 96-lane 4-tile v2 stack: lanes 0–23 → tile 0, 24–47 → tile 1, etc.
//!
//! [`TileDispatch`] is the pre-computed lookup table; [`LaneBinding`] is a single tile's
//! ownership window. Both are `Copy` (all fields are `usize`).

use crate::error::validate::Validate;

/// A single tile's lane ownership window — exclusive right bound `[lane_start, lane_end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneBinding {
    /// Tile index (0-based).
    pub tile: usize,
    /// First lane owned by this tile (inclusive).
    pub lane_start: usize,
    /// One past the last lane owned by this tile (exclusive).
    pub lane_end: usize,
}

impl LaneBinding {
    /// Number of lanes this tile owns.
    #[must_use]
    pub fn lane_count(&self) -> usize {
        self.lane_end - self.lane_start
    }
}

/// Maps `manifest.tx_nets[lane]` → tile index for a balanced equal-partition stack.
///
/// Constructed once per manifest via [`TileDispatch::new`]; the dispatch table is
/// immutable and `Clone`-able.
#[derive(Debug, Clone)]
pub struct TileDispatch {
    lanes: usize,
    tiles: usize,
    bindings: Vec<LaneBinding>,
}

impl TileDispatch {
    /// Build the equal-partition dispatch table for `lanes` total lanes across `tiles` tiles.
    ///
    /// # Errors
    ///
    /// * [`Validate::KwaversBeamStepContract`] — if `lanes == 0`, `tiles == 0`, or
    ///   `lanes % tiles != 0` (uneven partition).
    pub fn new(lanes: usize, tiles: usize) -> Result<Self, crate::Error> {
        if lanes == 0 || tiles == 0 {
            return Err(Validate::KwaversBeamStepContract(format!(
                "TileDispatch requires lanes > 0 and tiles > 0 (got lanes={lanes}, tiles={tiles})"
            ))
            .into());
        }
        if lanes % tiles != 0 {
            return Err(Validate::KwaversBeamStepContract(format!(
                "TileDispatch requires lanes ({lanes}) divisible by tiles ({tiles}); \
                 remainder = {}",
                lanes % tiles
            ))
            .into());
        }
        let per_tile = lanes / tiles;
        let bindings = (0..tiles)
            .map(|t| LaneBinding {
                tile: t,
                lane_start: t * per_tile,
                lane_end: (t + 1) * per_tile,
            })
            .collect();
        Ok(Self { lanes, tiles, bindings })
    }

    /// Borrow the full binding table (one entry per tile).
    #[must_use]
    pub fn bindings(&self) -> &[LaneBinding] {
        &self.bindings
    }

    /// Total lane count.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Total tile count.
    #[must_use]
    pub fn tiles(&self) -> usize {
        self.tiles
    }

    /// Which tile owns lane `lane`? Returns `None` if `lane >= self.lanes`.
    #[must_use]
    pub fn tile_for_lane(&self, lane: usize) -> Option<usize> {
        if lane >= self.lanes {
            return None;
        }
        let per_tile = self.lanes / self.tiles;
        Some(lane / per_tile)
    }
}
