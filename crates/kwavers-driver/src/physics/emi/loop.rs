//! Shoelace polygon area + first-order partial-inductance estimate for a planar commutation
//! loop (file on disk: `loop.rs`; the parent [`super`] declares it as `pub mod r#loop;`
//! because `loop` is a Rust reserved keyword).

use super::MU0;

use crate::geom::Point;

/// Shoelace area (mm²) of a polygon given in nm coordinates.
///
/// Slice-internal (`pub(super)`) — the scene walker in [`super::scene::commutation_loops`]
/// is the sole caller (the quadrilateral `[cap_a, ic_a, ic_b, cap_b]` is the canonical
/// input). Not re-exported from [`super`] so external code cannot bypass the
/// `CommutationLoop` shape to construct an arbitrary `area` value.
pub(super) fn polygon_area_mm2(poly: &[Point]) -> f64 {
    let n = poly.len();
    let mut acc: i128 = 0;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        acc += a.x.0 as i128 * b.y.0 as i128 - b.x.0 as i128 * a.y.0 as i128;
    }
    (acc.abs() as f64 / 2.0) * 1.0e-12 // nm² → mm²
}

/// First-order partial-inductance estimate (nH) for a planar loop of area `a_mm2`.
///
/// `L ≈ μ₀·√area` (dimensionally H·m/m): √(mm²)=mm=1e-3 m ⇒ nH scale. A 1 mm² loop ⇒
/// ~1.26 nH.
#[must_use]
pub fn loop_inductance_nh(a_mm2: f64) -> f64 {
    MU0 * (a_mm2 * 1.0e-6).sqrt() * 1.0e9
}
