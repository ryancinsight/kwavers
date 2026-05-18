use super::*;
use crate::domain::grid::Grid;
use crate::solver::forward::pstd::dg::{DGConfig, DgBoundaryCondition};
use std::sync::Arc;

fn make_solver(nx: usize, ny: usize, nz: usize) -> DGSolver {
    make_solver_with_order(nx, ny, nz, 1)
}

fn make_solver_with_order(nx: usize, ny: usize, nz: usize, polynomial_order: usize) -> DGSolver {
    make_solver_with_boundary(nx, ny, nz, polynomial_order, DgBoundaryCondition::Periodic)
}

fn make_solver_with_boundary(
    nx: usize,
    ny: usize,
    nz: usize,
    polynomial_order: usize,
    boundary_condition: DgBoundaryCondition,
) -> DGSolver {
    let grid = Arc::new(Grid::new(nx, ny, nz, 0.5, 0.75, 1.25).unwrap());
    let config = DGConfig {
        polynomial_order,
        sound_speed: 2.0,
        boundary_condition,
        ..DGConfig::default()
    };
    DGSolver::new(config, grid).unwrap()
}

fn weighted_rate(solver: &DGSolver, rhs: &Array3<f64>, var: usize) -> f64 {
    let topology = DgTopology::from_grid(&solver.grid, solver.n_nodes).unwrap();
    let mut rate = 0.0;
    for elem in 0..topology.n_elements {
        for node in 0..topology.nodes_per_element {
            rate += topology.node_weight(node, &solver.weights) * rhs[(elem, node, var)];
        }
    }
    rate
}

#[test]
fn constant_2d_acoustic_state_has_zero_rhs() {
    let solver = make_solver(4, 4, 1);
    let state = Array3::from_shape_fn(
        solver.acoustic_tensor_state_shape().unwrap(),
        |(_, _, v)| match v {
            ACOUSTIC_PRESSURE_VAR => 2.0,
            ACOUSTIC_VELOCITY_X_VAR => 0.25,
            ACOUSTIC_VELOCITY_Y_VAR => -0.5,
            _ => 0.0,
        },
    );
    let mut rhs = Array3::zeros(state.dim());

    solver
        .compute_acoustic_tensor_rhs_into(&state, 1.25, &mut rhs)
        .unwrap();

    for &value in &rhs {
        assert!(value.abs() < 1.0e-12);
    }
}

#[test]
fn tensor_2d_acoustic_rhs_preserves_component_masses() {
    let solver = make_solver(4, 4, 1);
    let state = Array3::from_shape_fn(
        solver.acoustic_tensor_state_shape().unwrap(),
        |(e, n, v)| (0.3 * e as f64 + 0.7 * n as f64 + 0.2 * v as f64).sin(),
    );
    let mut rhs = Array3::zeros(state.dim());

    solver
        .compute_acoustic_tensor_rhs_into(&state, 1.25, &mut rhs)
        .unwrap();

    for var in [
        ACOUSTIC_PRESSURE_VAR,
        ACOUSTIC_VELOCITY_X_VAR,
        ACOUSTIC_VELOCITY_Y_VAR,
    ] {
        let rate = weighted_rate(&solver, &rhs, var);
        assert!(
            rate.abs() < 1.0e-10,
            "2D acoustic DG must conserve weighted component mass for var {var}; rate={rate:e}"
        );
    }
}

#[test]
fn tensor_3d_acoustic_rhs_preserves_component_masses() {
    let solver = make_solver(4, 4, 4);
    let state = Array3::from_shape_fn(
        solver.acoustic_tensor_state_shape().unwrap(),
        |(e, n, v)| (0.2 * e as f64 + 0.5 * n as f64 + 0.11 * v as f64).cos(),
    );
    let mut rhs = Array3::zeros(state.dim());

    solver
        .compute_acoustic_tensor_rhs_into(&state, 1.25, &mut rhs)
        .unwrap();

    for var in 0..ACOUSTIC_VARIABLES {
        let rate = weighted_rate(&solver, &rhs, var);
        assert!(
            rate.abs() < 1.0e-10,
            "3D acoustic DG must conserve weighted component mass for var {var}; rate={rate:e}"
        );
    }
}

#[test]
fn tensor_absorbing_characteristic_boundary_preserves_outgoing_characteristic() {
    let impedance = 3.0;
    let amplitude = 2.0;
    let right_going = boundary::NormalState {
        pressure: amplitude,
        velocity_normal: amplitude / impedance,
    };
    let left_going = boundary::NormalState {
        pressure: amplitude,
        velocity_normal: -amplitude / impedance,
    };

    let preserved_right =
        boundary::absorbing_characteristic_external_state(right_going, impedance, true);
    assert!((preserved_right.pressure - right_going.pressure).abs() < 1.0e-12);
    assert!((preserved_right.velocity_normal - right_going.velocity_normal).abs() < 1.0e-12);

    let rejected_right =
        boundary::absorbing_characteristic_external_state(right_going, impedance, false);
    assert!(rejected_right.pressure.abs() < 1.0e-12);
    assert!(rejected_right.velocity_normal.abs() < 1.0e-12);

    let preserved_left =
        boundary::absorbing_characteristic_external_state(left_going, impedance, false);
    assert!((preserved_left.pressure - left_going.pressure).abs() < 1.0e-12);
    assert!((preserved_left.velocity_normal - left_going.velocity_normal).abs() < 1.0e-12);

    let rejected_left =
        boundary::absorbing_characteristic_external_state(left_going, impedance, true);
    assert!(rejected_left.pressure.abs() < 1.0e-12);
    assert!(rejected_left.velocity_normal.abs() < 1.0e-12);
}

#[test]
fn tensor_absorbing_boundary_replaces_periodic_exterior_face_state() {
    let periodic = make_solver_with_boundary(2, 1, 1, 1, DgBoundaryCondition::Periodic);
    let absorbing =
        make_solver_with_boundary(2, 1, 1, 1, DgBoundaryCondition::AbsorbingCharacteristic);
    let state = Array3::from_shape_fn(
        periodic.acoustic_tensor_state_shape().unwrap(),
        |(_, _, var)| {
            if var == ACOUSTIC_PRESSURE_VAR {
                1.0
            } else {
                0.0
            }
        },
    );
    let mut rhs_periodic = Array3::zeros(state.dim());
    let mut rhs_absorbing = Array3::zeros(state.dim());

    periodic
        .compute_acoustic_tensor_rhs_into(&state, 1.25, &mut rhs_periodic)
        .unwrap();
    absorbing
        .compute_acoustic_tensor_rhs_into(&state, 1.25, &mut rhs_absorbing)
        .unwrap();

    let periodic_norm = rhs_periodic.iter().map(|value| value.abs()).sum::<f64>();
    assert!(
        periodic_norm < 1.0e-12,
        "constant state must remain an exact periodic DG null mode; norm={periodic_norm:e}"
    );

    let absorbing_norm = rhs_absorbing.iter().map(|value| value.abs()).sum::<f64>();
    assert!(
        absorbing_norm > 1.0e-6,
        "absorbing boundary must replace periodic wraparound exterior states; norm={absorbing_norm:e}"
    );
}

#[test]
fn tensor_acoustic_step_preserves_constant_state() {
    let solver = make_solver(4, 4, 4);
    let mut state = Array3::from_elem(solver.acoustic_tensor_state_shape().unwrap(), 0.125_f64);
    let mut workspace = AcousticDgTensorWorkspace::new(state.dim());

    solver
        .step_acoustic_tensor_ssp_rk3(&mut state, 1.25, 0.01, &mut workspace)
        .unwrap();

    for &value in &state {
        assert!((value - 0.125).abs() < 1.0e-12);
    }
}

#[test]
fn tensor_uniform_projection_evaluates_gll_polynomial_on_grid_coordinates() {
    let solver = make_solver(4, 4, 1);
    let topology = DgTopology::from_grid(&solver.grid, solver.n_nodes).unwrap();
    let mut state = Array3::zeros(solver.acoustic_tensor_state_shape().unwrap());
    for elem in 0..topology.n_elements {
        let elem_coords = topology.element_coords(elem);
        for node in 0..topology.nodes_per_element {
            let node_coords = topology.node_coords(node);
            let x = dg_physical_coordinate(&solver, elem_coords[0], node_coords[0], 0);
            let y = dg_physical_coordinate(&solver, elem_coords[1], node_coords[1], 1);
            state[(elem, node, ACOUSTIC_PRESSURE_VAR)] = 1.0 + 2.0 * x - 0.5 * y;
        }
    }
    let mut pressure = Array3::zeros((solver.grid.nx, solver.grid.ny, solver.grid.nz));

    solver
        .project_acoustic_tensor_pressure_to_uniform_grid(&state, &mut pressure)
        .unwrap();

    for i in 0..solver.grid.nx {
        for j in 0..solver.grid.ny {
            let x = i as f64 * solver.grid.dx;
            let y = j as f64 * solver.grid.dy;
            let expected = 1.0 + 2.0 * x - 0.5 * y;
            let error = (pressure[(i, j, 0)] - expected).abs();
            assert!(
                error < 1.0e-12,
                "uniform projection must reproduce affine fields at ({i},{j}); error={error:e}"
            );
        }
    }
}

#[test]
fn tensor_weak_cell_source_weights_conserve_reference_cell_measure() {
    let solver = make_solver_with_order(8, 8, 1, 3);
    let topology = DgTopology::from_grid(&solver.grid, solver.n_nodes).unwrap();
    let sources = solver
        .acoustic_tensor_cell_source_weights([5, 6, 0])
        .unwrap();

    assert_eq!(sources.len(), topology.nodes_per_element);
    let reference_measure = sources.iter().fold(0.0, |sum, source| {
        sum + topology.node_weight(source.node, &solver.weights) * source.weight
    });
    let expected = (2.0 / solver.n_nodes as f64).powi(topology.active_dim as i32);
    assert!(
        (reference_measure - expected).abs() < 1.0e-12,
        "weak source weights must integrate one uniform cell; got {reference_measure:e}, expected {expected:e}"
    );
}

#[test]
fn tensor_ssp_rk3_source_callback_uses_stage_times() {
    let solver = make_solver(4, 4, 1);
    let mut state = Array3::zeros(solver.acoustic_tensor_state_shape().unwrap());
    let mut workspace = AcousticDgTensorWorkspace::new(state.dim());
    let t0 = 2.0;
    let dt = 0.1;

    solver
        .step_acoustic_tensor_ssp_rk3_with_source(
            &mut state,
            1.25,
            dt,
            &mut workspace,
            t0,
            |stage_t, rhs| {
                for elem in 0..rhs.shape()[0] {
                    for node in 0..rhs.shape()[1] {
                        rhs[(elem, node, ACOUSTIC_PRESSURE_VAR)] += stage_t;
                    }
                }
            },
        )
        .unwrap();

    let expected = t0 * dt + 0.5 * dt * dt;
    for &value in state
        .index_axis(ndarray::Axis(2), ACOUSTIC_PRESSURE_VAR)
        .iter()
    {
        assert!(
            (value - expected).abs() < 1.0e-12,
            "stage-time source integration mismatch: got {value:e}, expected {expected:e}"
        );
    }
}

fn dg_physical_coordinate(
    solver: &DGSolver,
    element_index: usize,
    node_index: usize,
    axis: usize,
) -> f64 {
    let spacing = [solver.grid.dx, solver.grid.dy, solver.grid.dz][axis];
    let element_span = solver.n_nodes as f64 * spacing;
    element_index as f64 * element_span + 0.5 * (solver.xi_nodes[node_index] + 1.0) * element_span
}
