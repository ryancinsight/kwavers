use super::*;

#[test]
fn cbs_solver_reports_decreasing_fixed_point_residual() {
    let grid = GridSpec::new((2, 1, 1), 0.01).unwrap();
    let source_density = [
        Complex64::new(1.0 / grid.cell_volume_m3(), 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let potential = [0.05, -0.02];
    let loose = solve_volume_field(
        grid,
        3.0,
        &potential,
        &source_density,
        CbsConfig {
            max_iterations: 1,
            relative_tolerance: 1.0e-14,
        },
    )
    .unwrap();
    let refined = solve_volume_field(
        grid,
        3.0,
        &potential,
        &source_density,
        CbsConfig {
            max_iterations: 8,
            relative_tolerance: 1.0e-14,
        },
    )
    .unwrap();

    assert!(refined.relative_residual < loose.relative_residual);
    assert!(refined.iterations >= loose.iterations);
}
