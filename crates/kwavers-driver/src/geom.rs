//! Exact board geometry.
//!
//! Board coordinates are **exact integers in nanometres** ([`Nm`]), matching KiCad's internal
//! representation. This is a correctness choice, not a precision tier: design-rule clearances are
//! exact distances, so floating-point coordinates would admit rounding errors at the very
//! boundaries the rules police. Costs (a search heuristic, not a measured quantity) are `f64`.
//!
//! # SSOT for [`Nm`]
//!
//! Phase 1a relocated the authoritative [`Nm`] (length newtype over integer nanometres) to
//! [`crate::units`]. `geom` re-exports it transparently so every existing `crate::geom::Nm`
//! reference — including the field types on [`Point`] / [`GridSpec`] / the board model — keeps
//! type-identical (no migration cost at the call-site). The authoritative path forward is
//! [`crate::units::Nm`]; `crate::geom::Nm` is kept as a convenience alias.

pub use crate::units::Nm;

/// A point on the board in exact nanometre coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Point {
    /// X coordinate.
    pub x: Nm,
    /// Y coordinate.
    pub y: Nm,
}

impl Point {
    /// Construct a point from two nanometre coordinates.
    #[must_use]
    pub fn new(x: Nm, y: Nm) -> Self {
        Point { x, y }
    }

    /// Manhattan (L1) distance to another point, in nanometres.
    #[must_use]
    pub fn manhattan(self, other: Point) -> Nm {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }

    /// Euclidean distance to another point, in nanometres (as `f64`).
    #[must_use]
    pub fn euclid(self, other: Point) -> f64 {
        let dx = (self.x.0 - other.x.0) as f64;
        let dy = (self.y.0 - other.y.0) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Signed orientation of the triplet `(a, b, c)` (twice the signed triangle area): `>0` left turn,
/// `<0` right turn, `0` collinear. Exact integer arithmetic (nm coordinates fit in `i128`).
#[must_use]
pub fn orient(a: Point, b: Point, c: Point) -> i128 {
    let (ax, ay) = (a.x.0 as i128, a.y.0 as i128);
    let (bx, by) = (b.x.0 as i128, b.y.0 as i128);
    let (cx, cy) = (c.x.0 as i128, c.y.0 as i128);
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Whether segments `p1p2` and `p3p4` *properly* cross (intersect at an interior point of both).
/// Shared endpoints and collinear overlap are not counted — this measures genuine lane crossings.
#[must_use]
pub fn segments_cross(p1: Point, p2: Point, p3: Point, p4: Point) -> bool {
    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);
    ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
}

/// Euclidean distance (nm, as `f64`) from point `p` to segment `ab` — the closest approach, with
/// the foot clamped to the segment endpoints. Degenerate (zero-length) `ab` reduces to point distance.
#[must_use]
pub fn dist_point_seg(p: Point, a: Point, b: Point) -> f64 {
    let (ax, ay) = (a.x.0 as f64, a.y.0 as f64);
    let (bx, by) = (b.x.0 as f64, b.y.0 as f64);
    let (px, py) = (p.x.0 as f64, p.y.0 as f64);
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return p.euclid(a);
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    let (cx, cy) = (ax + t * dx, ay + t * dy);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

/// Minimum Euclidean distance (nm, as `f64`) between segments `ab` and `cd`. Zero if they cross or
/// touch; otherwise the smallest of the four endpoint-to-opposite-segment distances (exact for the
/// non-crossing case, since the closest pair then involves at least one endpoint).
#[must_use]
pub fn dist_seg_seg(a: Point, b: Point, c: Point, d: Point) -> f64 {
    if segments_cross(a, b, c, d) {
        return 0.0;
    }
    dist_point_seg(a, c, d)
        .min(dist_point_seg(b, c, d))
        .min(dist_point_seg(c, a, b))
        .min(dist_point_seg(d, a, b))
}

/// Whether `p` lies inside `poly` using the even-odd rule. Boundary points are not treated as
/// inside; callers that need boundary-inclusive behavior should combine this with segment-distance
/// checks using the required tolerance.
#[must_use]
pub fn point_in_polygon(p: Point, poly: &[Point]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        if pi.y.0 == pj.y.0 {
            j = i;
            continue;
        }
        let edge_x = (pj.x.0 - pi.x.0) as f64 * (p.y.0 - pi.y.0) as f64 / (pj.y.0 - pi.y.0) as f64
            + pi.x.0 as f64;
        if ((pi.y.0 > p.y.0) != (pj.y.0 > p.y.0)) && ((p.x.0 as f64) < edge_x) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Minimum distance from `p` to the closed polygon boundary. Returns `None` for a degenerate
/// polygon with fewer than two points.
#[must_use]
pub fn distance_to_polygon_boundary(p: Point, poly: &[Point]) -> Option<f64> {
    if poly.len() < 2 {
        return None;
    }
    let mut best = f64::INFINITY;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        best = best.min(dist_point_seg(p, a, b));
    }
    Some(best)
}

/// Convex hull of a point set (Andrew's monotone chain), returned counter-clockwise without the
/// duplicated closing point. Fewer than 3 unique points returns the input sorted (degenerate hull).
/// Exact integer orientation, so collinear points on an edge are dropped.
#[must_use]
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut pts: Vec<Point> = points.to_vec();
    pts.sort_unstable_by_key(|p| (p.x.0, p.y.0));
    pts.dedup_by(|a, b| a.x.0 == b.x.0 && a.y.0 == b.y.0);
    let n = pts.len();
    if n < 3 {
        return pts;
    }
    let mut hull: Vec<Point> = Vec::with_capacity(2 * n);
    // Lower hull, then upper hull; a right turn (orient <= 0) pops the middle point.
    for &p in pts.iter() {
        while hull.len() >= 2 && orient(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower && orient(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop(); // last point == first
    hull
}

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
        let nx = (width.0 / pitch.0).max(0) as usize + 1;
        let ny = (height.0 / pitch.0).max(0) as usize + 1;
        if nx == 0 || ny == 0 || nlayers == 0 || pitch.0 <= 0 {
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
    #[must_use]
    pub fn cell_of(&self, p: Point) -> (usize, usize) {
        let rel_x = (p.x - self.origin.x).0;
        let rel_y = (p.y - self.origin.y).0;
        let ix = ((rel_x + self.pitch.0 / 2) / self.pitch.0).clamp(0, self.nx as i64 - 1);
        let iy = ((rel_y + self.pitch.0 / 2) / self.pitch.0).clamp(0, self.ny as i64 - 1);
        (ix as usize, iy as usize)
    }

    /// Board-space centre of a cell.
    #[must_use]
    pub fn point_of(&self, ix: usize, iy: usize) -> Point {
        Point::new(
            self.origin.x + Nm(self.pitch.0 * ix as i64),
            self.origin.y + Nm(self.pitch.0 * iy as i64),
        )
    }

    /// Flatten `(ix, iy, layer)` to a node index. Layout is layer-major so that all nodes of a
    /// layer are contiguous (cache-friendly plane sweeps).
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
    #[must_use]
    pub fn node_coords(&self, node: usize) -> (usize, usize, usize) {
        let layer = node / (self.nx * self.ny);
        let rem = node % (self.nx * self.ny);
        (rem % self.nx, rem / self.nx, layer)
    }
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
