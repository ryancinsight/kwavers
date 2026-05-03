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
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "barycentric coords must sum to 1"
    );
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
