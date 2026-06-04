use super::*;
use kwavers_grid::Grid;
use crate::forward::pstd::dg::config::{DGConfig, ShockCaptureConfig, WenoDegree};
use crate::forward::pstd::dg::flux::LimiterType;
use ndarray::Array3;
use std::sync::Arc;

fn weighted_mass_rate(solver: &DGSolver, rhs: &Array3<f64>) -> f64 {
    let (n_elements, n_nodes, n_vars) = rhs.dim();
    let mut rate = 0.0;
    match solver.coefficient_layout {
        CoefficientLayout::TensorProduct(topology) if n_nodes == topology.nodes_per_element => {
            for elem in 0..n_elements {
                for node in 0..n_nodes {
                    let weight = topology.node_weight(node, &solver.weights);
                    for var in 0..n_vars {
                        rate += weight * rhs[(elem, node, var)];
                    }
                }
            }
        }
        _ => {
            for elem in 0..n_elements {
                for node in 0..n_nodes {
                    for var in 0..n_vars {
                        rate += solver.weights[node] * rhs[(elem, node, var)];
                    }
                }
            }
        }
    }
    rate
}

#[test]
fn dg_rhs_preserves_periodic_global_mass() {
    let grid = Arc::new(Grid::new(12, 2, 2, 1.0, 1.0, 1.0).unwrap());
    let config = DGConfig {
        polynomial_order: 2,
        sound_speed: 1.0,
        ..Default::default()
    };
    let mut solver = DGSolver::new(config, grid).unwrap();
    solver.initialize_modal_coefficients(4, 1);
    {
        let coeffs = solver.modal_coefficients_mut().unwrap();
        for elem in 0..4 {
            for node in 0..3 {
                coeffs[(elem, node, 0)] =
                    (0.7 * elem as f64 + 0.3 * node as f64).sin() + 0.2 * elem as f64;
            }
        }
    }

    let rhs = solver
        .compute_rhs_from_coeffs(solver.modal_coefficients.as_ref().unwrap(), 1.0)
        .unwrap();
    let rate = weighted_mass_rate(&solver, &rhs);

    assert!(
        rate.abs() < 1.0e-12,
        "periodic DG advection must conserve weighted global mass; rate={rate:e}"
    );
}

#[test]
fn dg_tensor_rhs_preserves_periodic_global_mass() {
    let grid = Arc::new(Grid::new(4, 4, 2, 0.25, 0.5, 0.75).unwrap());
    let config = DGConfig {
        polynomial_order: 1,
        sound_speed: 1.3,
        ..Default::default()
    };
    let mut solver = DGSolver::new(config, Arc::clone(&grid)).unwrap();
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                field[(i, j, k)] = (0.4 * i as f64 + 0.7 * j as f64 + 0.2 * k as f64).sin()
                    + 0.05 * (i * j + k) as f64;
            }
        }
    }
    solver.project_to_dg(&field).unwrap();

    let rhs = solver
        .compute_rhs_from_coeffs(
            solver.modal_coefficients.as_ref().unwrap(),
            config.sound_speed,
        )
        .unwrap();
    let rate = weighted_mass_rate(&solver, &rhs);
    let tolerance = f64::EPSILON * rhs.len() as f64 * rhs.iter().map(|v| v.abs()).sum::<f64>();

    assert!(
        rate.abs() <= tolerance,
        "tensor DG periodic advection must conserve weighted global mass; rate={rate:e}, tolerance={tolerance:e}"
    );
}

#[test]
fn solve_step_reuses_rk_workspace_and_preserves_constant_state() {
    let grid = Arc::new(Grid::new(6, 3, 3, 1.0, 1.0, 1.0).unwrap());
    let config = DGConfig {
        polynomial_order: 2,
        sound_speed: 1.0,
        ..Default::default()
    };
    let mut solver = DGSolver::new(config, grid).unwrap();
    let mut field = Array3::from_elem((6, 3, 3), 2.0);

    solver.solve_step(&mut field, 0.01).unwrap();
    let coeff_ptr = solver.modal_coefficients.as_ref().unwrap().as_ptr();
    let original_ptr = solver.rk_original.as_ptr();
    let stage_ptr = solver.rk_stage.as_ptr();
    let rhs_ptr = solver.rk_rhs.as_ptr();

    solver.solve_step(&mut field, 0.01).unwrap();

    assert_eq!(
        solver.modal_coefficients.as_ref().unwrap().as_ptr(),
        coeff_ptr
    );
    assert_eq!(solver.rk_original.as_ptr(), original_ptr);
    assert_eq!(solver.rk_stage.as_ptr(), stage_ptr);
    assert_eq!(solver.rk_rhs.as_ptr(), rhs_ptr);
    let rhs = solver
        .compute_rhs_from_coeffs(solver.modal_coefficients.as_ref().unwrap(), 1.0)
        .unwrap();
    for &value in &rhs {
        assert!(value.abs() < 1e-12);
    }
    for &value in solver.modal_coefficients.as_ref().unwrap() {
        assert!((value - 2.0).abs() < 1e-12);
    }
}

#[test]
fn solve_step_applies_configured_shock_capture_and_preserves_dg_mass_mean() {
    let grid = Arc::new(Grid::new(10, 2, 2, 1.0, 1.0, 1.0).unwrap());
    let config = DGConfig {
        polynomial_order: 2,
        limiter_type: LimiterType::Minmod,
        shock_capture: ShockCaptureConfig {
            enabled: true,
            limiter: WenoDegree::Weno3,
            threshold: 0.05,
            apply_per_stage: true,
        },
        sound_speed: 1.0,
        ..Default::default()
    };
    let mut solver = DGSolver::new(config, grid).unwrap();
    solver.initialize_modal_coefficients(5, 1);
    {
        let coeffs = solver.modal_coefficients_mut().unwrap();
        coeffs[(0, 0, 0)] = 1.0;
        coeffs[(0, 1, 0)] = 1.0;
        coeffs[(0, 2, 0)] = 1.0;
        coeffs[(1, 0, 0)] = 5.0;
        coeffs[(1, 1, 0)] = 5.0;
        coeffs[(1, 2, 0)] = 5.0;
        coeffs[(2, 0, 0)] = 9.0;
        coeffs[(2, 1, 0)] = 1.0;
        coeffs[(2, 2, 0)] = 9.0;
        coeffs[(3, 0, 0)] = 5.0;
        coeffs[(3, 1, 0)] = 5.0;
        coeffs[(3, 2, 0)] = 5.0;
        coeffs[(4, 0, 0)] = 1.0;
        coeffs[(4, 1, 0)] = 1.0;
        coeffs[(4, 2, 0)] = 1.0;
    }
    let weights = solver.weights.clone();
    let mass_mean_before = ((0..solver.n_nodes)
        .map(|node| weights[node] * solver.modal_coefficients.as_ref().unwrap()[(2, node, 0)])
        .sum::<f64>())
        / weights.iter().sum::<f64>();
    let coeff_ptr = solver.modal_coefficients.as_ref().unwrap().as_ptr();
    let mut field = Array3::zeros((10, 2, 2));

    solver.solve_step(&mut field, 0.0).unwrap();

    let coeffs = solver.modal_coefficients.as_ref().unwrap();
    assert_eq!(coeffs.as_ptr(), coeff_ptr);
    let mass_mean_after = ((0..solver.n_nodes)
        .map(|node| weights[node] * coeffs[(2, node, 0)])
        .sum::<f64>())
        / weights.iter().sum::<f64>();
    assert!((mass_mean_after - mass_mean_before).abs() < 1.0e-12);
    for node in 0..solver.n_nodes {
        assert!((coeffs[(2, node, 0)] - mass_mean_before).abs() < 1.0e-12);
    }
}

#[test]
fn solve_step_leaves_modal_oscillation_when_shock_capture_disabled() {
    let grid = Arc::new(Grid::new(10, 2, 2, 1.0, 1.0, 1.0).unwrap());
    let config = DGConfig {
        polynomial_order: 1,
        shock_capture: ShockCaptureConfig {
            enabled: false,
            ..Default::default()
        },
        sound_speed: 1.0,
        ..Default::default()
    };
    let mut solver = DGSolver::new(config, grid).unwrap();
    solver.initialize_modal_coefficients(5, 1);
    {
        let coeffs = solver.modal_coefficients_mut().unwrap();
        coeffs[(2, 0, 0)] = 7.0;
        coeffs[(2, 1, 0)] = 3.0;
    }
    let mut field = Array3::zeros((10, 2, 2));

    solver.solve_step(&mut field, 0.0).unwrap();

    let coeffs = solver.modal_coefficients.as_ref().unwrap();
    assert!((coeffs[(2, 0, 0)] - 7.0).abs() < 1e-12);
    assert!((coeffs[(2, 1, 0)] - 3.0).abs() < 1e-12);
}
