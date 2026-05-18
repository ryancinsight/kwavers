use super::*;
use crate::domain::medium::HomogeneousMedium;
use crate::KwaversError;

fn config(spatial_order: usize) -> ThermalDiffusionConfig {
    ThermalDiffusionConfig {
        enable_bioheat: false,
        enable_hyperbolic: false,
        track_thermal_dose: false,
        spatial_order,
        ..Default::default()
    }
}

#[test]
fn second_order_laplacian_keeps_singleton_axis_active_for_other_axes() {
    let grid = Grid::new(5, 5, 1, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    let field = Array3::from_shape_fn((5, 5, 1), |(i, j, _)| (i * i) as f64 + 2.0 * (j * j) as f64);
    solver.set_temperature(field);

    solver.calculate_laplacian(&grid).unwrap();

    assert_eq!(solver.laplacian_workspace[[2, 2, 0]], 6.0);
    assert_eq!(solver.laplacian_workspace[[0, 2, 0]], 0.0);
}

#[test]
fn fourth_order_laplacian_falls_back_per_axis_on_narrow_dimensions() {
    let grid = Grid::new(7, 3, 1, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(4), &grid);
    let field = Array3::from_shape_fn((7, 3, 1), |(i, j, _)| (i * i) as f64 + 3.0 * (j * j) as f64);
    solver.set_temperature(field);

    solver.calculate_laplacian(&grid).unwrap();

    let error = (solver.laplacian_workspace[[3, 1, 0]] - 8.0).abs();
    assert!(error <= 8.0 * f64::EPSILON);
}

#[test]
fn fourth_order_laplacian_is_exact_for_quadratic_field_in_3d() {
    let grid = Grid::new(7, 7, 7, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(4), &grid);
    let field = Array3::from_shape_fn((7, 7, 7), |(i, j, k)| {
        let x = i as f64 - 3.0;
        let y = j as f64 - 3.0;
        let z = k as f64 - 3.0;
        x * x + 2.0 * y * y + 3.0 * z * z
    });
    solver.set_temperature(field);

    solver.calculate_laplacian(&grid).unwrap();

    let error = (solver.laplacian_workspace[[3, 3, 3]] - 12.0).abs();
    assert!(error <= 12.0 * f64::EPSILON);
}

#[test]
fn unsupported_spatial_order_returns_validation_error_without_mutation() {
    let grid = Grid::new(5, 5, 1, 1.0, 1.0, 1.0).unwrap();
    let mut solver = ThermalDiffusionSolver::new(config(6), &grid);

    let err = solver.calculate_laplacian(&grid).unwrap_err();

    assert!(
        matches!(err, KwaversError::Validation(_)),
        "unsupported spatial order must return a validation error, got {err:?}"
    );
    assert_eq!(solver.config.spatial_order, 6);
    assert_eq!(solver.laplacian_workspace[[2, 2, 0]], 0.0);
}

#[test]
fn standard_update_consumes_borrowed_source_view_without_source_clone() {
    let grid = Grid::new(3, 3, 1, 1.0, 1.0, 1.0).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let mut solver = ThermalDiffusionSolver::new(config(2), &grid);
    solver.set_temperature(Array3::from_elem((3, 3, 1), 310.0));
    let mut source = Array3::zeros((3, 3, 1));
    source[[1, 1, 0]] = 5.0;

    solver
        .update(&medium, &grid, 2.0, Some(source.view()))
        .unwrap();

    assert_eq!(solver.temperature()[[1, 1, 0]], 320.0);
    assert_eq!(solver.temperature()[[0, 0, 0]], 310.0);
}
