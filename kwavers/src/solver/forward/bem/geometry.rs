//! Geometry utilities for BEM boundary element computations.
//!
//! Provides 3D vector operations, triangle metrics, barycentric coordinate
//! computation, and point-to-triangle distance calculation used throughout
//! the BEM solver for element integration and near-field detection.
//!
//! # Algorithms
//!
//! - **Barycentric coordinates**: Cramer's rule on the edge-vector Gram matrix
//!   (Möller & Trumbore 1997, §2.1).
//! - **Point-to-triangle distance**: Voronoi region closest-point test
//!   (Ericson 2004, *Real-Time Collision Detection*, §5.1.5).
//!
//! # References
//!
//! - Ericson, C. (2004). *Real-Time Collision Detection*. Morgan Kaufmann, §5.1.5.
//! - Möller, T. & Trumbore, B. (1997). "Fast, minimum storage ray-triangle
//!   intersection." *J. Graphics Tools* 2(1):21–28.

// ── 3D vector primitives ─────────────────────────────────────────────────────

/// 3D dot product: a · b.
#[inline]
pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    // FMA-friendly: compiler will fuse on targets with FMA support.
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3D vector subtraction: a − b.
#[inline]
pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 3D vector addition: a + b.
#[inline]
pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scalar multiplication: s · a.
#[inline]
pub(crate) fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// 3D cross product: a × b.
#[inline]
pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Squared Euclidean norm: ‖a‖².
#[inline]
pub(crate) fn norm_sq(a: [f64; 3]) -> f64 {
    dot(a, a)
}

// ── Triangle metrics ─────────────────────────────────────────────────────────

/// Maximum edge length of a triangle (characteristic element size).
///
/// Returns max(‖p2−p1‖, ‖p3−p2‖, ‖p1−p3‖). Used for near-field / far-field
/// classification in adaptive quadrature: when `distance / h_max < threshold`,
/// the element is considered near-field and requires refined integration.
///
/// # Performance
///
/// Compares squared edge lengths and takes a single square root of the
/// maximum, avoiding two unnecessary `sqrt` calls.
pub(crate) fn triangle_characteristic_length(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let d12_sq = norm_sq(sub(p2, p1));
    let d23_sq = norm_sq(sub(p3, p2));
    let d31_sq = norm_sq(sub(p1, p3));
    d12_sq.max(d23_sq).max(d31_sq).sqrt()
}

/// Triangle area and outward unit normal from vertices (p1, p2, p3).
///
/// Computes `n = (p2 − p1) × (p3 − p1)` and returns `(area, n̂)` where
/// `area = ‖n‖ / 2` and `n̂ = n / ‖n‖`.
///
/// For degenerate triangles (area < ε), returns `(0, [0, 0, 1])`.
pub(crate) fn triangle_area_normal(
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
) -> (f64, [f64; 3]) {
    let n = cross(sub(p2, p1), sub(p3, p1));
    let norm = norm_sq(n).sqrt();
    if norm < 1e-30 {
        (0.0, [0.0, 0.0, 1.0])
    } else {
        (0.5 * norm, scale(n, 1.0 / norm))
    }
}

// ── Barycentric coordinates ──────────────────────────────────────────────────

/// Barycentric coordinates of point `p` with respect to triangle (p1, p2, p3).
///
/// # Algorithm (Cramer's rule on edge-vector Gram matrix)
///
/// Given edge vectors e0 = p2 − p1, e1 = p3 − p1, and d = p − p1, the
/// barycentric coordinates (u, v, w) satisfy:
///
/// ```text
/// ⎡ e0·e0  e0·e1 ⎤ ⎡ v ⎤   ⎡ e0·d ⎤
/// ⎣ e0·e1  e1·e1 ⎦ ⎣ w ⎦ = ⎣ e1·d ⎦
///
/// Δ = (e0·e0)(e1·e1) − (e0·e1)²
/// v = [(e1·e1)(e0·d) − (e0·e1)(e1·d)] / Δ
/// w = [(e0·e0)(e1·d) − (e0·e1)(e0·d)] / Δ
/// u = 1 − v − w
/// ```
///
/// Returns `[u, v, w]` such that `p ≈ u·p1 + v·p2 + w·p3` and `u + v + w = 1`.
/// For degenerate triangles (Δ ≈ 0), returns `[1, 0, 0]`.
///
/// # Reference
///
/// Möller, T. & Trumbore, B. (1997). *J. Graphics Tools* 2(1):21–28.
pub(crate) fn barycentric_coords(
    p: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
) -> [f64; 3] {
    let e0 = sub(p2, p1);
    let e1 = sub(p3, p1);
    let d = sub(p, p1);
    let d00 = dot(e0, e0);
    let d01 = dot(e0, e1);
    let d02 = dot(e0, d);
    let d11 = dot(e1, e1);
    let d12 = dot(e1, d);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-18 {
        [1.0, 0.0, 0.0]
    } else {
        let inv = 1.0 / denom;
        let v = (d11 * d02 - d01 * d12) * inv;
        let w = (d00 * d12 - d01 * d02) * inv;
        [1.0 - v - w, v, w]
    }
}

// ── Point-to-triangle distance ───────────────────────────────────────────────

/// Minimum Euclidean distance from point `p` to triangle (p1, p2, p3).
///
/// # Algorithm (Ericson 2004, §5.1.5)
///
/// Tests all seven Voronoi regions of the triangle in order:
///
/// | Region   | Condition                              | Closest feature |
/// |----------|----------------------------------------|-----------------|
/// | V(p1)    | d1 ≤ 0 ∧ d2 ≤ 0                       | Vertex p1       |
/// | V(p2)    | d3 ≥ 0 ∧ d4 ≤ d3                      | Vertex p2       |
/// | V(p3)    | d6 ≥ 0 ∧ d5 ≤ d6                      | Vertex p3       |
/// | E(p1p2)  | vc ≤ 0 ∧ d1 ≥ 0 ∧ d3 ≤ 0             | Edge p1–p2      |
/// | E(p1p3)  | vb ≤ 0 ∧ d2 ≥ 0 ∧ d6 ≤ 0             | Edge p1–p3      |
/// | E(p2p3)  | va ≤ 0 ∧ (d4−d3) ≥ 0 ∧ (d5−d6) ≥ 0   | Edge p2–p3      |
/// | Interior | Otherwise                              | Face interior   |
///
/// where d1..d6 are dot products of edge vectors with vertex-to-point vectors,
/// and va, vb, vc are signed area ratios (cross-product tests).
///
/// # Complexity
///
/// O(1) — constant number of dot products and comparisons.
///
/// # Reference
///
/// Ericson, C. (2004). *Real-Time Collision Detection*. Morgan Kaufmann, §5.1.5.
pub(crate) fn point_to_triangle_distance(
    p: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
) -> f64 {
    let ab = sub(p2, p1);
    let ac = sub(p3, p1);
    let ap = sub(p, p1);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);

    // Region V(p1): closest to vertex p1
    if d1 <= 0.0 && d2 <= 0.0 {
        return norm_sq(ap).sqrt();
    }

    let bp = sub(p, p2);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);

    // Region V(p2): closest to vertex p2
    if d3 >= 0.0 && d4 <= d3 {
        return norm_sq(bp).sqrt();
    }

    // Region E(p1p2): closest to edge p1–p2
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let proj = add(p1, scale(ab, v));
        return norm_sq(sub(p, proj)).sqrt();
    }

    let cp = sub(p, p3);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);

    // Region V(p3): closest to vertex p3
    if d6 >= 0.0 && d5 <= d6 {
        return norm_sq(cp).sqrt();
    }

    // Region E(p1p3): closest to edge p1–p3
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let proj = add(p1, scale(ac, w));
        return norm_sq(sub(p, proj)).sqrt();
    }

    // Region E(p2p3): closest to edge p2–p3
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let proj = add(p2, scale(sub(p3, p2), w));
        return norm_sq(sub(p, proj)).sqrt();
    }

    // Region Interior: closest point on triangle face
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let proj = add(p1, add(scale(ab, v), scale(ac, w)));
    norm_sq(sub(p, proj)).sqrt()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    // ── Vector primitives ────────────────────────────────────────────────

    #[test]
    fn test_dot_product() {
        assert!((dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])).abs() < EPS);
        assert!((dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) - 32.0).abs() < EPS);
        assert!((dot([3.0, 0.0, 0.0], [3.0, 0.0, 0.0]) - 9.0).abs() < EPS);
    }

    #[test]
    fn test_cross_product() {
        // x × y = z
        let c = cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < EPS);
        assert!((c[1]).abs() < EPS);
        assert!((c[2] - 1.0).abs() < EPS);

        // Anti-commutativity: a × b = −(b × a)
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let ab = cross(a, b);
        let ba = cross(b, a);
        for i in 0..3 {
            assert!((ab[i] + ba[i]).abs() < EPS);
        }
    }

    // ── Triangle metrics ─────────────────────────────────────────────────

    #[test]
    fn test_triangle_characteristic_length() {
        // Right triangle with legs 3 and 4, hypotenuse 5
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [3.0, 0.0, 0.0];
        let p3 = [0.0, 4.0, 0.0];
        let h = triangle_characteristic_length(p1, p2, p3);
        assert!((h - 5.0).abs() < EPS);
    }

    #[test]
    fn test_triangle_area_normal_unit() {
        // Unit right triangle in the xy-plane
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        let (area, normal) = triangle_area_normal(p1, p2, p3);
        assert!((area - 0.5).abs() < EPS);
        assert!((normal[0]).abs() < EPS);
        assert!((normal[1]).abs() < EPS);
        assert!((normal[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_triangle_area_normal_degenerate() {
        // Degenerate triangle (all points coincident)
        let p = [1.0, 2.0, 3.0];
        let (area, _normal) = triangle_area_normal(p, p, p);
        assert!(area.abs() < EPS);
    }

    // ── Barycentric coordinates ──────────────────────────────────────────

    #[test]
    fn test_barycentric_at_vertices() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];

        let b1 = barycentric_coords(p1, p1, p2, p3);
        assert!((b1[0] - 1.0).abs() < EPS);
        assert!((b1[1]).abs() < EPS);
        assert!((b1[2]).abs() < EPS);

        let b2 = barycentric_coords(p2, p1, p2, p3);
        assert!((b2[0]).abs() < EPS);
        assert!((b2[1] - 1.0).abs() < EPS);
        assert!((b2[2]).abs() < EPS);

        let b3 = barycentric_coords(p3, p1, p2, p3);
        assert!((b3[0]).abs() < EPS);
        assert!((b3[1]).abs() < EPS);
        assert!((b3[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_barycentric_centroid() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [3.0, 0.0, 0.0];
        let p3 = [0.0, 3.0, 0.0];
        let centroid = [1.0, 1.0, 0.0];
        let b = barycentric_coords(centroid, p1, p2, p3);
        for &bi in &b {
            assert!((bi - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_barycentric_partition_of_unity() {
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 2.0, 0.0];
        let p3 = [0.0, 0.0, 3.0];
        let p = [0.3, 0.5, 0.7];
        let b = barycentric_coords(p, p1, p2, p3);
        let sum: f64 = b.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "barycentric coords must sum to 1");
    }

    // ── Point-to-triangle distance ───────────────────────────────────────

    #[test]
    fn test_distance_point_above_centroid() {
        // Point directly above the centroid of a flat triangle in the xy-plane
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        let p = [0.25, 0.25, 1.0]; // above interior
        let d = point_to_triangle_distance(p, p1, p2, p3);
        assert!((d - 1.0).abs() < 1e-10, "distance should be 1.0, got {d}");
    }

    #[test]
    fn test_distance_at_vertex() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        // Point 2 units from vertex p1
        let p = [-1.0, -1.0, 0.0];
        let d = point_to_triangle_distance(p, p1, p2, p3);
        let expected = 2.0_f64.sqrt();
        assert!(
            (d - expected).abs() < 1e-10,
            "distance to vertex: expected {expected}, got {d}"
        );
    }

    #[test]
    fn test_distance_on_edge() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [2.0, 0.0, 0.0];
        let p3 = [0.0, 2.0, 0.0];
        // Point 3 units directly below the midpoint of edge p1–p2
        let p = [1.0, 0.0, -3.0];
        let d = point_to_triangle_distance(p, p1, p2, p3);
        assert!((d - 3.0).abs() < 1e-10, "distance to edge: got {d}");
    }

    #[test]
    fn test_distance_zero_on_surface() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        // Point on the triangle surface
        let p = [0.25, 0.25, 0.0];
        let d = point_to_triangle_distance(p, p1, p2, p3);
        assert!(d < 1e-10, "distance on surface should be ~0, got {d}");
    }
}
