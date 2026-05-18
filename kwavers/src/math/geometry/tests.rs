use super::{distance3, make_circle, make_line, normalize3, *};

#[test]
fn test_make_disc_basic() {
    let grid = (32, 32, 1);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let center = [1.6e-3, 1.6e-3, 0.0];
    let radius = 0.5e-3;

    let mask = make_disc(grid, spacing, center, radius).unwrap();

    assert_eq!(mask.dim(), (32, 32, 1));
    assert!(mask[[16, 16, 0]]);

    let count = mask.iter().filter(|&&x| x).count();
    let expected_area = std::f64::consts::PI * radius * radius;
    let cell_area = spacing.0 * spacing.1;
    let expected_count = (expected_area / cell_area).round() as usize;

    assert!((count as f64 - expected_count as f64).abs() / (expected_count as f64) < 0.2);
}

#[test]
fn test_make_disc_invalid_radius() {
    let grid = (32, 32, 1);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let center = [1.6e-3, 1.6e-3, 0.0];

    let result = make_disc(grid, spacing, center, -1.0);
    assert!(result.is_err());

    let result = make_disc(grid, spacing, center, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_make_ball_basic() {
    let grid = (32, 32, 32);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let center = [1.6e-3, 1.6e-3, 1.6e-3];
    let radius = 0.5e-3;

    let mask = make_ball(grid, spacing, center, radius).unwrap();

    assert_eq!(mask.dim(), (32, 32, 32));
    assert!(mask[[16, 16, 16]]);

    let count = mask.iter().filter(|&&x| x).count();
    let expected_volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    let cell_volume = spacing.0 * spacing.1 * spacing.2;
    let expected_count = (expected_volume / cell_volume).round() as usize;

    assert!((count as f64 - expected_count as f64).abs() / (expected_count as f64) < 0.25);
}

#[test]
fn test_make_ball_invalid_radius() {
    let grid = (32, 32, 32);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let center = [1.6e-3, 1.6e-3, 1.6e-3];

    let result = make_ball(grid, spacing, center, -1.0);
    assert!(result.is_err());
}

#[test]
fn test_make_sphere_alias() {
    let grid = (32, 32, 32);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let center = [1.6e-3, 1.6e-3, 1.6e-3];
    let radius = 0.5e-3;

    let ball = make_ball(grid, spacing, center, radius).unwrap();
    let sphere = make_sphere(grid, spacing, center, radius).unwrap();

    assert_eq!(ball, sphere);
}

#[test]
fn test_make_line_diagonal() {
    let grid = (32, 32, 32);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let start = [0.0, 0.0, 0.0];
    let end = [3.1e-3, 3.1e-3, 3.1e-3];

    let mask = make_line(grid, spacing, start, end).unwrap();

    assert_eq!(mask.dim(), (32, 32, 32));
    assert!(mask[[0, 0, 0]]);
    assert!(mask[[31, 31, 31]]);

    let count = mask.iter().filter(|&&x| x).count();
    assert!(count > 31 && count < 60);
}

#[test]
fn test_make_line_axis_aligned() {
    let grid = (32, 32, 1);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let start = [0.0, 1.5e-3, 0.0];
    let end = [3.1e-3, 1.5e-3, 0.0];

    let mask = make_line(grid, spacing, start, end).unwrap();

    for i in 0..32 {
        assert!(mask[[i, 15, 0]]);
    }
}

#[test]
fn test_disc_symmetry() {
    let grid = (64, 64, 1);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let center = [3.2e-3, 3.2e-3, 0.0];
    let radius = 1.0e-3;

    let mask = make_disc(grid, spacing, center, radius).unwrap();

    for r in 1..5 {
        assert_eq!(mask[[32 + r, 32, 0]], mask[[32 - r, 32, 0]]);
        assert_eq!(mask[[32, 32 + r, 0]], mask[[32, 32 - r, 0]]);
    }
}

#[test]
fn test_ball_symmetry() {
    let grid = (64, 64, 64);
    let spacing = (0.1e-3, 0.1e-3, 0.1e-3);
    let center = [3.2e-3, 3.2e-3, 3.2e-3];
    let radius = 1.0e-3;

    let mask = make_ball(grid, spacing, center, radius).unwrap();

    for r in 1..5 {
        assert_eq!(mask[[32 + r, 32, 32]], mask[[32 - r, 32, 32]]);
        assert_eq!(mask[[32, 32 + r, 32]], mask[[32, 32 - r, 32]]);
        assert_eq!(mask[[32, 32, 32 + r]], mask[[32, 32, 32 - r]]);
    }
}

// ─── make_circle exact pixel tests ───────────────────────────────────────────

/// Origin-centered circle at integer radius on unit grid: center pixel is off, perimeter on.
///
/// Grid 7×7×1, spacing=1.0 m, center=[3.0,3.0,0.0], radius=2.0, thickness=1.
/// Points at distance |2−r| ≤ 0.5 are set true; center (dist=0) is false.
#[test]
fn circle_center_is_off_perimeter_is_on() {
    let dim = (7, 7, 1);
    let spacing = (1.0, 1.0, 1.0);
    let center = [3.0, 3.0, 0.0];
    let radius = 2.0_f64;

    let mask = make_circle(dim, spacing, center, radius, 1).expect("make_circle should succeed");

    // Center must be false
    assert!(!mask[[3, 3, 0]], "center must be false (inside circle)");

    // Axis-aligned perimeter points at Manhattan distance 2 must be true
    // Point [1,3,0]: dist from center = 2.0 → on perimeter
    assert!(mask[[1, 3, 0]], "[1,3,0] at distance 2.0 must be true");
    assert!(mask[[5, 3, 0]], "[5,3,0] at distance 2.0 must be true");
    assert!(mask[[3, 1, 0]], "[3,1,0] at distance 2.0 must be true");
    assert!(mask[[3, 5, 0]], "[3,5,0] at distance 2.0 must be true");
}

/// `make_circle` rejects radius ≤ 0.
#[test]
fn circle_rejects_non_positive_radius() {
    assert!(make_circle((5, 5, 1), (1.0, 1.0, 1.0), [2.0, 2.0, 0.0], 0.0, 1).is_err());
    assert!(make_circle((5, 5, 1), (1.0, 1.0, 1.0), [2.0, 2.0, 0.0], -1.0, 1).is_err());
}

/// `make_circle` rejects thickness = 0.
#[test]
fn circle_rejects_zero_thickness() {
    assert!(make_circle((5, 5, 1), (1.0, 1.0, 1.0), [2.0, 2.0, 0.0], 1.0, 0).is_err());
}

// ─── distance3 exact tests ────────────────────────────────────────────────────

/// `distance3` of a 3-4-5 right triangle gives exactly 5.0.
///
/// a=[0,0,0], b=[3,4,0]: d = √(9+16) = √25 = 5.0.
#[test]
fn geometry_distance3_pythagorean_triple_exact() {
    let d = distance3([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
    assert!(
        (d - 5.0).abs() < 1e-14,
        "distance3([0,0,0],[3,4,0]) = {d} (expected 5.0)"
    );
}

/// `distance3` of coincident points is exactly 0.0.
#[test]
fn geometry_distance3_coincident_is_zero() {
    let d = distance3([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
    assert!(
        d.abs() < 1e-15,
        "distance3 of same point = {d} (expected 0.0)"
    );
}

/// `distance3` with 3D components: (1,2,2) → √(1+4+4) = √9 = 3.0.
#[test]
fn geometry_distance3_three_dimensional_exact() {
    let d = distance3([0.0, 0.0, 0.0], [1.0, 2.0, 2.0]);
    assert!(
        (d - 3.0).abs() < 1e-14,
        "distance3([0,0,0],[1,2,2]) = {d} (expected 3.0)"
    );
}

// ─── normalize3 exact tests ───────────────────────────────────────────────────

/// `normalize3` of [3,4,0] gives [0.6, 0.8, 0].
///
/// magnitude = √(9+16) = 5. [3/5, 4/5, 0] = [0.6, 0.8, 0].
#[test]
fn geometry_normalize3_pythagorean_exact() {
    let n = normalize3([3.0, 4.0, 0.0]);
    assert!((n[0] - 0.6).abs() < 1e-14, "n[0]={} (expected 0.6)", n[0]);
    assert!((n[1] - 0.8).abs() < 1e-14, "n[1]={} (expected 0.8)", n[1]);
    assert!(n[2].abs() < 1e-14, "n[2]={} (expected 0.0)", n[2]);
}

/// `normalize3` of a unit axis vector returns the same vector.
#[test]
fn geometry_normalize3_unit_vector_unchanged() {
    let n = normalize3([0.0, 0.0, 1.0]);
    assert!(n[0].abs() < 1e-15);
    assert!(n[1].abs() < 1e-15);
    assert!((n[2] - 1.0).abs() < 1e-15);
}

/// `normalize3` of a zero vector returns the zero vector.
#[test]
fn geometry_normalize3_zero_vector_returns_zero() {
    let n = normalize3([0.0, 0.0, 0.0]);
    assert!(
        n[0].abs() < 1e-15 && n[1].abs() < 1e-15 && n[2].abs() < 1e-15,
        "normalize3 of zero must return zero; got {:?}",
        n
    );
}

// ─── make_line exact pixel tests ─────────────────────────────────────────────

/// `make_line` on a 1-cell degenerate line (start=end) marks exactly one voxel.
#[test]
fn make_line_single_point_marks_one_voxel() {
    let dim = (5, 5, 5);
    let spacing = (1.0, 1.0, 1.0);
    let start = [2.0, 2.0, 2.0];
    let mask = make_line(dim, spacing, start, start).unwrap();
    let count = mask.iter().filter(|&&v| v).count();
    assert_eq!(
        count, 1,
        "degenerate line must mark exactly 1 voxel, got {count}"
    );
    assert!(
        mask[[2, 2, 2]],
        "the single marked voxel must be at start=end=[2,2,2]"
    );
}
