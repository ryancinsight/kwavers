use super::*;

/// Boundary condition enforcement:
/// a non-zero bubble concentration fixes the wall node exactly after a step.
/// # Panics
/// - Panics if `valid diffusion step should solve the tridiagonal system`.
///
#[test]
fn test_dirichlet_boundary_condition() {
    let solver = RadicalDiffusionSolver::new(10e-6);
    let mut concentrations = solver.zero_concentrations(1);
    let bubble_boundary = vec![1e-6_f64];
    let diffusion_coefficients = vec![2e-9_f64];

    let result = solver
        .step(
            &mut concentrations,
            &bubble_boundary,
            1e-9,
            &diffusion_coefficients,
        )
        .expect("valid diffusion step should solve the tridiagonal system");

    assert_eq!(concentrations[0][0], bubble_boundary[0]);
    assert_eq!(result.concentrations[0][0], bubble_boundary[0]);
    assert!(result.max_delta >= bubble_boundary[0]);
}

/// Far-field decay:
/// the outer Dirichlet boundary remains zero after repeated diffusion steps.
/// # Panics
/// - Panics if `valid diffusion step should preserve far-field boundary`.
///
#[test]
fn test_far_field_remains_zero() {
    let solver = RadicalDiffusionSolver::new(10e-6);
    let n = solver.n_points;
    let mut concentrations = solver.zero_concentrations(1);
    concentrations[0][n / 2] = 1e-6;

    let bubble_boundary = vec![0.0_f64];
    let diffusion_coefficients = vec![2e-9_f64];

    for _ in 0..100 {
        solver
            .step(
                &mut concentrations,
                &bubble_boundary,
                1e-9,
                &diffusion_coefficients,
            )
            .expect("valid diffusion step should preserve far-field boundary");
    }

    assert_eq!(concentrations[0][n - 1], 0.0);
}

/// Positivity:
/// a non-negative step-function initial condition remains non-negative.
/// # Panics
/// - Panics if `valid diffusion step should keep concentrations non-negative`.
///
#[test]
fn test_concentrations_non_negative() {
    let solver = RadicalDiffusionSolver::new(10e-6);
    let n = solver.n_points;
    let mut concentrations = solver.zero_concentrations(1);
    for concentration in concentrations[0].iter_mut().take(n / 2) {
        *concentration = 1e-6;
    }

    let bubble_boundary = vec![1e-6_f64];
    let diffusion_coefficients = vec![2e-9_f64];

    for _ in 0..50 {
        solver
            .step(
                &mut concentrations,
                &bubble_boundary,
                1e-9,
                &diffusion_coefficients,
            )
            .expect("valid diffusion step should keep concentrations non-negative");
    }

    for concentration in &concentrations[0] {
        assert!(
            *concentration >= 0.0,
            "negative concentration: {concentration:.4e}"
        );
    }
}

/// Grid invariant:
/// logarithmic radius nodes are strictly increasing and start at the bubble wall.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_radial_grid_strictly_increasing() {
    let solver = RadicalDiffusionSolver::new(5e-6);
    let grid = solver.radial_grid();

    for window in grid.windows(2) {
        assert!(window[1] > window[0]);
    }
    assert!((grid[0] - solver.r_bubble_m).abs() < 1e-18);
}
