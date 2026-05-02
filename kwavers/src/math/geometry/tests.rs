use super::*;

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
