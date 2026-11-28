//! Exact board geometry.
//!
//! Board coordinates are **exact integers in nanometres** ([`Nm`]), matching KiCad's internal
//! representation. This is a correctness choice, not a precision tier: design-rule clearances are
//! exact distances, so floating-point coordinates would admit rounding errors at the very
//! boundaries the rules police. Costs (a search heuristic, not a measured quantity) are `f64`.
//!
//! # SSOT for [`Nm`]
//!
//! The authoritative [`Nm`] — the length newtype over integer nanometres — is defined in
//! [`crate::units`]. `geom` re-exports it transparently, so `crate::geom::Nm` and
//! [`crate::units::Nm`] name the same type and every existing reference, including the
//! field types on [`Point`] / [`GridSpec`] / the board model, stays type-identical. Prefer
//! [`crate::units::Nm`] in new code.

pub use electra_board::geom::*;

/// The discrete routing grid: a uniform lattice of square cells stacked over `nlayers` copper
/// layers. It is the shared vocabulary mapping exact [`Point`]s to flat node indices, used by
/// both the routing grid and the cost field (so neither depends on the other).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSpec {
    /// Number of cells along X.
    pub nx: usize,
    /// Number of cells along Y.
    pub ny: usize,
    /// Number of copper layers.
    pub nlayers: usize,
    /// Cell pitch (cell side length).
    pub pitch: Nm,
    /// Board-space coordinate of cell (0, 0)'s centre.
    pub origin: Point,
}

impl GridSpec {
    /// Build a grid spec covering `[0, width] x [0, height]` with the given pitch and layer count.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Geometry::EmptyGrid`] when `pitch ≤ 0`, `nlayers == 0`, or the
    /// computed cell count on either axis is zero.
    pub fn cover(width: Nm, height: Nm, pitch: Nm, nlayers: usize) -> crate::Result<Self> {
        if pitch.0 <= 0 || nlayers == 0 {
            return Err(crate::error::Geometry::EmptyGrid.into());
        }
        let nx = (width.0 / pitch.0).max(0) as usize + 1;
        let ny = (height.0 / pitch.0).max(0) as usize + 1;
        if nx == 0 || ny == 0 {
            return Err(crate::error::Geometry::EmptyGrid.into());
        }
        Ok(GridSpec {
            nx,
            ny,
            nlayers,
            pitch,
            origin: Point::new(Nm(0), Nm(0)),
        })
    }

    /// Total number of grid nodes (`nx * ny * nlayers`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nlayers
    }

    /// Whether the grid has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Nearest in-plane cell `(ix, iy)` to a board point, clamped to grid bounds.
    #[inline]
    #[must_use]
    pub fn cell_of(&self, p: Point) -> (usize, usize) {
        let rel_x = (p.x - self.origin.x).0;
        let rel_y = (p.y - self.origin.y).0;
        let ix = ((rel_x + self.pitch.0 / 2) / self.pitch.0).clamp(0, self.nx as i64 - 1);
        let iy = ((rel_y + self.pitch.0 / 2) / self.pitch.0).clamp(0, self.ny as i64 - 1);
        (ix as usize, iy as usize)
    }

    /// Board-space centre of a cell.
    #[inline]
    #[must_use]
    pub fn point_of(&self, ix: usize, iy: usize) -> Point {
        Point::new(
            self.origin.x + Nm(self.pitch.0 * ix as i64),
            self.origin.y + Nm(self.pitch.0 * iy as i64),
        )
    }

    /// Flatten `(ix, iy, layer)` to a node index. Layout is layer-major so that all nodes of a
    /// layer are contiguous (cache-friendly plane sweeps).
    #[inline]
    #[must_use]
    pub fn node_index(&self, ix: usize, iy: usize, layer: usize) -> usize {
        (layer * self.ny + iy) * self.nx + ix
    }

    /// In-plane cells `(ix, iy)` whose centre lies within the axis-aligned rectangle centred at
    /// `center` with half-extents `(half_w, half_h)`. Used to inflate a pad into a clearance halo.
    #[must_use]
    pub fn cells_in_rect(&self, center: Point, half_w: Nm, half_h: Nm) -> Vec<(usize, usize)> {
        let lo = self.cell_of(Point::new(center.x - half_w, center.y - half_h));
        let hi = self.cell_of(Point::new(center.x + half_w, center.y + half_h));
        let mut out = Vec::new();
        for iy in lo.1..=hi.1 {
            for ix in lo.0..=hi.0 {
                let p = self.point_of(ix, iy);
                if (p.x - center.x).abs() <= half_w && (p.y - center.y).abs() <= half_h {
                    out.push((ix, iy));
                }
            }
        }
        out
    }

    /// Inverse of [`GridSpec::node_index`].
    #[inline]
    #[must_use]
    pub fn node_coords(&self, node: usize) -> (usize, usize, usize) {
        let layer = node / (self.nx * self.ny);
        let rem = node % (self.nx * self.ny);
        (rem % self.nx, rem / self.nx, layer)
    }
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
/// board. Single source of truth shared by [`crate::io::pcb_emit::write_kicad_pcb`] (emission) and
/// the router keepout reservation ([`crate::pipeline`] / `crate::place::energy`), so the holes the
/// router avoids are exactly the holes that get drilled.
#[must_use]
pub fn mechanical_features(w_mm: f64, h_mm: f64) -> Vec<MechFeature> {
    let mi = (w_mm.min(h_mm) * 0.09).clamp(3.0, 8.0);
    let fi = mi + 3.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nm_mm_roundtrip_is_exact_at_micron() {
        // 1.27 mm is exactly representable in nm; round-trip must be bit-exact.
        assert_eq!(Nm::from_mm(1.27), Nm(1_270_000));
        assert_eq!(Nm(1_270_000).to_mm(), 1.27);
    }

    #[test]
    fn cell_index_roundtrip() {
        let g = GridSpec::cover(Nm::from_mm(10.0), Nm::from_mm(5.0), Nm::from_mm(0.5), 4).unwrap();
        for layer in 0..g.nlayers {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let n = g.node_index(ix, iy, layer);
                    assert_eq!(g.node_coords(n), (ix, iy, layer));
                    // The centre of a cell must map back to that cell.
                    assert_eq!(g.cell_of(g.point_of(ix, iy)), (ix, iy));
                }
            }
        }
        assert_eq!(g.len(), g.nx * g.ny * g.nlayers);
    }

    #[test]
    fn cell_of_clamps_out_of_bounds() {
        let g = GridSpec::cover(Nm::from_mm(2.0), Nm::from_mm(2.0), Nm::from_mm(1.0), 1).unwrap();
        let far = Point::new(Nm::from_mm(99.0), Nm::from_mm(-99.0));
        let (ix, iy) = g.cell_of(far);
        assert!(ix < g.nx && iy < g.ny);
        assert_eq!((ix, iy), (g.nx - 1, 0));
    }
}
