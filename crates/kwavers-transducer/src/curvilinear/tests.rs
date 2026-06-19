//! Value-semantic tests for the convex (curvilinear) array geometry.
//! Expected values are derived analytically from the closed-form geometry.

use super::ConvexArrayGeometry;

const R_C: f64 = 60e-3; // 60 mm radius of curvature (typical abdominal probe)

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

#[test]
fn all_elements_lie_on_the_curvature_arc() {
    let geo = ConvexArrayGeometry::from_total_angle(R_C, 9, 0.8).expect("geo");
    let c = geo.curvature_center();
    for i in 0..geo.num_elements() {
        let r = dist(geo.element_position(i), c);
        assert!((r - R_C).abs() < 1e-12, "element {i} off the arc: {r} vs {R_C}");
    }
}

#[test]
fn center_element_is_at_apex_facing_z() {
    // Odd count ⇒ middle element at θ=0: apex (0,0,0), normal +z.
    let geo = ConvexArrayGeometry::from_angular_pitch(R_C, 7, 0.01).expect("geo");
    let mid = 3;
    assert!((geo.element_angle(mid)).abs() < 1e-15);
    let p = geo.element_position(mid);
    assert!(norm(p) < 1e-12, "apex must be at origin, got {p:?}");
    let n = geo.element_normal(mid);
    assert!((n[0]).abs() < 1e-15 && (n[1]).abs() < 1e-15 && (n[2] - 1.0).abs() < 1e-15);
}

#[test]
fn normals_are_unit_radial_and_tangents_orthogonal() {
    let geo = ConvexArrayGeometry::from_arc_pitch(R_C, 12, 0.5e-3).expect("geo");
    let c = geo.curvature_center();
    for i in 0..geo.num_elements() {
        let n = geo.element_normal(i);
        assert!((norm(n) - 1.0).abs() < 1e-12, "normal {i} not unit");
        // Radial: normal == (position − center)/R_c.
        let p = geo.element_position(i);
        let radial = [(p[0] - c[0]) / R_C, (p[1] - c[1]) / R_C, (p[2] - c[2]) / R_C];
        for k in 0..3 {
            assert!((n[k] - radial[k]).abs() < 1e-12, "normal {i} not radial on axis {k}");
        }
        // Tangent ⊥ normal.
        let t = geo.element_tangent(i);
        let dot = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
        assert!(dot.abs() < 1e-12, "tangent {i} not orthogonal to normal: {dot}");
    }
}

#[test]
fn arc_pitch_round_trips_and_layout_is_symmetric() {
    let pitch = 0.45e-3;
    let geo = ConvexArrayGeometry::from_arc_pitch(R_C, 10, pitch).expect("geo");
    assert!((geo.arc_pitch() - pitch).abs() < 1e-15, "arc pitch should round-trip");
    // Symmetric about the apex: θ_0 == −θ_{N−1}.
    let n = geo.num_elements();
    assert!((geo.element_angle(0) + geo.element_angle(n - 1)).abs() < 1e-12);
}

#[test]
fn aperture_width_matches_chord_formula() {
    let geo = ConvexArrayGeometry::from_total_angle(R_C, 5, 1.0).expect("geo");
    // span = (N−1)·Δθ = total angle = 1.0 rad; chord = 2 R_c sin(span/2).
    let expected = 2.0 * R_C * (1.0_f64 / 2.0).sin();
    assert!((geo.aperture_width() - expected).abs() < 1e-12);
    assert!((geo.total_angular_span() - 1.0).abs() < 1e-12);
}

#[test]
fn focusing_at_curvature_center_gives_zero_relative_delays() {
    // Every element is exactly R_c from the centre of curvature, so focusing
    // there requires identical (zero relative) delays — exact analytical check.
    let geo = ConvexArrayGeometry::from_total_angle(R_C, 8, 0.9).expect("geo");
    let delays = geo.focusing_delays(geo.curvature_center(), 1540.0).expect("delays");
    for (i, &d) in delays.iter().enumerate() {
        assert!(d.abs() < 1e-15, "delay {i} should be 0 focusing at curvature center, got {d}");
    }
}

#[test]
fn focusing_delays_are_nonnegative_symmetric_and_min_zero() {
    // On-axis focus in front of the apex: delays symmetric about the centre,
    // all ≥ 0, and at least one is 0 (the farthest element).
    let geo = ConvexArrayGeometry::from_total_angle(R_C, 9, 0.8).expect("geo");
    let focus = [0.0, 0.0, 40e-3]; // 40 mm deep, on axis
    let delays = geo.focusing_delays(focus, 1540.0).expect("delays");
    let n = geo.num_elements();
    for &d in &delays {
        assert!(d >= 0.0, "delays must be non-negative");
    }
    assert!(delays.iter().copied().fold(f64::INFINITY, f64::min).abs() < 1e-18, "min delay must be 0");
    for i in 0..n {
        assert!(
            (delays[i] - delays[n - 1 - i]).abs() < 1e-15,
            "on-axis focus delays must be symmetric: {} vs {}",
            delays[i],
            delays[n - 1 - i]
        );
    }
}

#[test]
fn rejects_invalid_parameters() {
    assert!(ConvexArrayGeometry::from_angular_pitch(-1.0, 4, 0.1).is_err());
    assert!(ConvexArrayGeometry::from_angular_pitch(R_C, 0, 0.1).is_err());
    assert!(ConvexArrayGeometry::from_angular_pitch(R_C, 4, 0.0).is_err());
    assert!(ConvexArrayGeometry::from_arc_pitch(R_C, 4, -1.0).is_err());
    assert!(ConvexArrayGeometry::from_total_angle(R_C, 1, 0.5).is_err());
    let geo = ConvexArrayGeometry::from_total_angle(R_C, 4, 0.5).expect("geo");
    assert!(geo.focusing_delays([0.0, 0.0, 0.03], 0.0).is_err());
}
