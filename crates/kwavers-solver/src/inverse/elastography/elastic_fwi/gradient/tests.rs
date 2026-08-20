use super::*;
use crate::forward::elastic::swe::ElasticWaveConfig;
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;

#[test]
fn streamed_kernel_matches_full_history_oracle() {
    let grid = Grid::new(10, 10, 10, 1.0e-3, 1.0e-3, 1.0e-3).expect("grid");
    let medium =
        HomogeneousMedium::elastic_homogeneous(1000.0, 3.464_101_6, 2.0, &grid).expect("medium");
    let config = ElasticWaveConfig {
        pml_thickness: 2,
        ..ElasticWaveConfig::default()
    };
    let solver = ElasticWaveSolver::new(&grid, &medium, config).expect("solver");
    let dt = solver.recommended_timestep(0.3);
    let n_steps = 24;

    let mut forward_force = ElasticPointForce::zeros((4, 5, 5), n_steps);
    forward_force.fx[0] = 4.0e6;
    forward_force.fy[2] = -7.0e6;
    forward_force.fz[5] = 3.0e6;
    let forward_full = solver
        .propagate_point_forces(n_steps, dt, &[forward_force.clone()])
        .expect("full forward history");
    let forward = solver
        .propagate_point_force_displacements(n_steps, dt, &[forward_force])
        .expect("forward history");

    let mut adjoint_force = ElasticPointForce::zeros((5, 5, 5), n_steps);
    adjoint_force.fx[1] = -2.0e6;
    adjoint_force.fy[4] = 5.0e6;
    adjoint_force.fz[7] = 6.0e6;
    let adjoint_forces = [adjoint_force];
    let adjoint = solver
        .propagate_point_forces(n_steps, dt, &adjoint_forces)
        .expect("adjoint history");

    let expected = k_mu_kernel_from_histories(&forward_full, &adjoint, dt, grid.spacing());
    let actual = stream_k_mu_kernel(
        &solver,
        &forward,
        &adjoint_forces,
        dt,
        grid.spacing(),
        false,
    )
    .expect("streamed kernel");

    assert!(expected.0.iter().any(|value| *value != 0.0));
    assert!(expected.1.iter().any(|value| *value > 0.0));
    assert_eq!(actual.0, expected.0, "streamed gradient");
    assert_eq!(actual.1, expected.1, "streamed illumination");
}

#[test]
fn plane_strain_kernel_matches_full_history_oracle() {
    let grid = Grid::new(11, 10, 1, 1.0e-3, 1.3e-3, 2.0e-3).expect("grid");
    let medium =
        HomogeneousMedium::elastic_homogeneous(1000.0, 3.464_101_6, 2.0, &grid).expect("medium");
    let solver = ElasticWaveSolver::new(
        &grid,
        &medium,
        ElasticWaveConfig {
            pml_thickness: 2,
            ..ElasticWaveConfig::default()
        },
    )
    .expect("solver");
    let dt = solver.recommended_timestep(0.3);
    let n_steps = 18;
    let mut forward_force = ElasticPointForce::zeros((4, 5, 0), n_steps);
    forward_force.fx[0] = 4.0e6;
    forward_force.fy[2] = -7.0e6;
    let forward_full = solver
        .propagate_point_forces(n_steps, dt, &[forward_force.clone()])
        .expect("full forward history");
    let forward = solver
        .propagate_point_force_displacements(n_steps, dt, &[forward_force])
        .expect("forward history");
    let mut adjoint_force = ElasticPointForce::zeros((5, 5, 0), n_steps);
    adjoint_force.fx[1] = -2.0e6;
    adjoint_force.fy[4] = 5.0e6;
    let adjoint_forces = [adjoint_force];
    let adjoint = solver
        .propagate_point_forces(n_steps, dt, &adjoint_forces)
        .expect("adjoint history");

    let expected = k_mu_kernel_from_histories(&forward_full, &adjoint, dt, grid.spacing());
    let actual = stream_k_mu_kernel(&solver, &forward, &adjoint_forces, dt, grid.spacing(), true)
        .expect("plane-strain kernel");

    assert!(expected.0.iter().any(|value| *value != 0.0));
    assert_eq!(actual.0, expected.0);
    assert_eq!(actual.1, expected.1);
}
