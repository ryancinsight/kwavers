use super::super::config::NewtonKrylovConfig;
use super::super::residual_metric::norm;
use super::super::state_vector::sorted_field_keys;
use super::*;
use crate::core::error::{KwaversError, ValidationError};
use crate::domain::field::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::solver::integration::nonlinear::GMRESConfig;
use ndarray::Array3;
use std::collections::HashMap;

fn single_field(
    grid: &Grid,
    field_type: UnifiedFieldType,
    value: f64,
) -> HashMap<UnifiedFieldType, Array3<f64>> {
    let mut fields = HashMap::new();
    fields.insert(field_type, Array3::from_elem(grid.dimensions(), value));
    fields
}

#[test]
fn test_monolithic_coupler_creation() {
    let newton_config = NewtonKrylovConfig::default();
    let gmres_config = GMRESConfig::default();
    let coupler = MonolithicCoupler::new(newton_config, gmres_config);

    assert!(coupler.convergence_history().is_empty());
    assert_eq!(coupler.physics_components.len(), 0);
}

/// Repeated shape-compatible solves reuse the previous-state snapshot buffer.
///
/// With constant pressure/temperature fields and acoustic absorption disabled,
/// `nabla^2 p = nabla^2 T = 0` and the thermal acoustic-heating term is zero.
/// The implicit residual is therefore exactly `u - u_prev`, so each solve
/// converges without changing the flattened state.  This isolates the memory
/// contract: `u_prev_scratch` must be refreshed by value while retaining its
/// allocation across calls.
#[test]
fn test_previous_state_snapshot_workspace_reuses_buffer() {
    let newton_config = NewtonKrylovConfig {
        max_newton_iterations: 2,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(newton_config, GMRESConfig::default());
    coupler.physics_coefficients.acoustic_absorption = 0.0;

    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();
    let mut fields = HashMap::new();
    fields.insert(
        UnifiedFieldType::Pressure,
        Array3::from_elem(grid.dimensions(), 1.0),
    );
    fields.insert(
        UnifiedFieldType::Temperature,
        Array3::from_elem(grid.dimensions(), 3.0),
    );

    let first = coupler
        .solve_coupled_step(&mut fields, 1e-6, &grid)
        .unwrap();
    assert!(first.converged);
    assert_eq!(first.final_residual, 0.0);

    let (nx, ny, nz) = grid.dimensions();
    let first_ptr = coupler.u_prev_scratch.as_ref().unwrap().as_ptr();
    assert_eq!(
        coupler.u_prev_scratch.as_ref().unwrap().dim(),
        (2 * nx, ny, nz)
    );

    fields
        .get_mut(&UnifiedFieldType::Pressure)
        .unwrap()
        .fill(2.0);
    fields
        .get_mut(&UnifiedFieldType::Temperature)
        .unwrap()
        .fill(4.0);

    let second = coupler
        .solve_coupled_step(&mut fields, 1e-6, &grid)
        .unwrap();
    assert!(second.converged);
    assert_eq!(second.final_residual, 0.0);

    let snapshot = coupler.u_prev_scratch.as_ref().unwrap();
    assert_eq!(snapshot.as_ptr(), first_ptr);

    let order = sorted_field_keys(&fields);
    let pressure_block = order
        .iter()
        .position(|&field_type| field_type == UnifiedFieldType::Pressure)
        .unwrap();
    let temperature_block = order
        .iter()
        .position(|&field_type| field_type == UnifiedFieldType::Temperature)
        .unwrap();

    assert_eq!(snapshot[[pressure_block * nx + 1, 1, 1]], 2.0);
    assert_eq!(snapshot[[temperature_block * nx + 1, 1, 1]], 4.0);
}

/// Non-converged Newton iterations reuse the `-F(u)` GMRES RHS buffer.
///
/// For a constant fluence field, `nabla^2 I = 0` and
/// `F_I = I - I_prev - dt(-mu_a I) = dt * mu_a * I`.  The GMRES RHS must
/// therefore be `-dt * mu_a * I` at every voxel and must retain its allocation
/// across repeated solves with the same flattened shape.
#[test]
fn test_newton_rhs_workspace_reuses_buffer_and_refreshes_values() {
    let newton_config = NewtonKrylovConfig {
        max_newton_iterations: 1,
        newton_tolerance: f64::MIN_POSITIVE,
        adaptive_step_size: false,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(newton_config, GMRESConfig::default());
    let mu_a = coupler.physics_coefficients.optical_absorption;
    let dt = 1e-6;
    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();
    let mut fields = HashMap::new();
    fields.insert(
        UnifiedFieldType::LightFluence,
        Array3::from_elem(grid.dimensions(), 5.0),
    );

    let first = coupler.solve_coupled_step(&mut fields, dt, &grid).unwrap();
    assert!(!first.converged);

    let rhs = coupler.rhs_scratch.as_ref().unwrap();
    let first_ptr = rhs.as_ptr();
    let expected_first = -dt * mu_a * 5.0;
    assert!(rhs
        .iter()
        .all(|&value| (value - expected_first).abs() < 1e-15));

    fields
        .get_mut(&UnifiedFieldType::LightFluence)
        .unwrap()
        .fill(7.0);
    let second = coupler.solve_coupled_step(&mut fields, dt, &grid).unwrap();
    assert!(!second.converged);

    let rhs = coupler.rhs_scratch.as_ref().unwrap();
    let expected_second = -dt * mu_a * 7.0;
    assert_eq!(rhs.as_ptr(), first_ptr);
    assert!(rhs
        .iter()
        .all(|&value| (value - expected_second).abs() < 1e-15));
}

/// Invalid timesteps are rejected before solver history or workspaces mutate.
#[test]
fn test_solve_rejects_nonpositive_dt_before_state_mutation() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    coupler.convergence_history.push(42.0);
    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();
    let mut fields = single_field(&grid, UnifiedFieldType::Pressure, 1.0);

    let error = coupler
        .solve_coupled_step(&mut fields, 0.0, &grid)
        .unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => {
            assert_eq!(parameter, "dt");
            assert_eq!(value, 0.0);
            assert_eq!(reason, "must be finite and positive");
        }
        other => panic!("expected timestep validation error, got {other:?}"),
    }
    assert_eq!(coupler.convergence_history(), &[42.0]);
    assert!(coupler.u_prev_scratch.is_none());
}

/// Empty field maps do not fabricate a one-cell monolithic state.
#[test]
fn test_solve_rejects_empty_field_set_before_flattening() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();
    let mut fields = HashMap::new();

    let error = coupler
        .solve_coupled_step(&mut fields, 1e-6, &grid)
        .unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::MissingField { field }) => {
            assert_eq!(field, "monolithic fields");
        }
        other => panic!("expected missing-field validation error, got {other:?}"),
    }
    assert!(coupler.convergence_history().is_empty());
}

/// Field volumes must match the grid before the stacked-state copy runs.
#[test]
fn test_solve_rejects_field_grid_shape_mismatch() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();
    let mut fields = HashMap::new();
    fields.insert(UnifiedFieldType::Pressure, Array3::zeros((2, 3, 3)));

    let error = coupler
        .solve_coupled_step(&mut fields, 1e-6, &grid)
        .unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::DimensionMismatch { expected, actual }) => {
            assert_eq!(expected, "(3, 3, 3)");
            assert_eq!(actual, "Pressure (2, 3, 3)");
        }
        other => panic!("expected dimension validation error, got {other:?}"),
    }
    assert!(coupler.u_prev_scratch.is_none());
}

/// Newton configuration errors are rejected before residual evaluation.
#[test]
fn test_solve_rejects_invalid_newton_contracts() {
    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();

    let zero_iter_config = NewtonKrylovConfig {
        max_newton_iterations: 0,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(zero_iter_config, GMRESConfig::default());
    let mut fields = single_field(&grid, UnifiedFieldType::Density, 1.0);
    let error = coupler
        .solve_coupled_step(&mut fields, 1e-6, &grid)
        .unwrap_err();
    match error {
        KwaversError::Validation(ValidationError::InvalidParameter { parameter, reason }) => {
            assert_eq!(parameter, "NewtonKrylovConfig::max_newton_iterations");
            assert_eq!(reason, "must be greater than zero");
        }
        other => panic!("expected Newton iteration validation error, got {other:?}"),
    }

    let zero_tolerance_config = NewtonKrylovConfig {
        newton_tolerance: 0.0,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(zero_tolerance_config, GMRESConfig::default());
    let mut fields = single_field(&grid, UnifiedFieldType::Density, 1.0);
    let error = coupler
        .solve_coupled_step(&mut fields, 1e-6, &grid)
        .unwrap_err();
    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => {
            assert_eq!(parameter, "NewtonKrylovConfig::newton_tolerance");
            assert_eq!(value, 0.0);
            assert_eq!(reason, "must be finite and positive");
        }
        other => panic!("expected Newton tolerance validation error, got {other:?}"),
    }
}

/// Invalid physical coefficients are rejected before residual divisions run.
#[test]
fn test_solve_rejects_invalid_physics_coefficients_before_residual() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    coupler.physics_coefficients.density = 0.0;
    let grid = Grid::new(3, 3, 3, 1e-3, 1e-3, 1e-3).unwrap();
    let mut fields = single_field(&grid, UnifiedFieldType::Temperature, 1.0);

    let error = coupler
        .solve_coupled_step(&mut fields, 1e-6, &grid)
        .unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => {
            assert_eq!(parameter, "PhysicsCoefficients::density");
            assert_eq!(value, 0.0);
            assert_eq!(reason, "must be finite and positive");
        }
        other => panic!("expected physics-coefficient validation error, got {other:?}"),
    }
    assert!(coupler.convergence_history().is_empty());
    assert!(coupler.u_prev_scratch.is_none());
}

/// Backtracking line search reuses the candidate-state workspace.
///
/// The `Density` block has no physics rate in the monolithic residual, so
/// `F(u) = u - u_prev`.  With `u_prev = 0` and `du = -u/2`, the first
/// candidate at `alpha = 1` gives `F(u + du) = u/2`, which satisfies the
/// sufficient-decrease test exactly.  The workspace must therefore contain the
/// half-state candidate and keep the same allocation across repeated calls.
#[test]
fn test_line_search_trial_workspace_reuses_buffer_and_refreshes_values() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);

    let u = Array3::from_elem(dims, 2.0);
    let du = Array3::from_elem(dims, -1.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    assert_eq!(alpha, 1.0);

    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    let first_ptr = trial.as_ptr();
    assert!(trial.iter().all(|&value| (value - 1.0).abs() < 1e-15));

    let u = Array3::from_elem(dims, 4.0);
    let du = Array3::from_elem(dims, -2.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    assert_eq!(alpha, 1.0);

    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    assert_eq!(trial.as_ptr(), first_ptr);
    assert!(trial.iter().all(|&value| (value - 2.0).abs() < 1e-15));
}

/// Adaptive line search uses the configured maximum trial step.
///
/// For the identity residual `F(u)=u-u_prev`, `u=2`, and `du=-2`, a configured
/// `alpha_max=0.5` gives the accepted candidate `u+alpha*du = 1`.  Returning
/// `1.0` here would prove the configuration field is dead solver state.
#[test]
fn test_line_search_uses_configured_initial_alpha() {
    let config = NewtonKrylovConfig {
        line_search_parameter: 0.5,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(config, GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let u = Array3::from_elem(dims, 2.0);
    let du = Array3::from_elem(dims, -2.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    assert_eq!(alpha, 0.5);
    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    assert!(trial.iter().all(|&value| (value - 1.0).abs() < 1e-15));
}

/// Invalid line-search step bounds are rejected before residual evaluation.
#[test]
fn test_line_search_rejects_invalid_configured_alpha() {
    let config = NewtonKrylovConfig {
        line_search_parameter: 0.0,
        ..NewtonKrylovConfig::default()
    };
    let mut coupler = MonolithicCoupler::new(config, GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let u = Array3::from_elem(dims, 2.0);
    let du = Array3::from_elem(dims, -1.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let error = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => {
            assert_eq!(parameter, "NewtonKrylovConfig::line_search_parameter");
            assert_eq!(value, 0.0);
            assert_eq!(reason, "must be finite and in (0, 1]");
        }
        other => panic!("expected line-search validation error, got {other:?}"),
    }
}

/// Failed backtracking returns the final evaluated step, not an untested step.
///
/// For the identity residual `F(u)=u-u_prev`, `u=1`, and `du=1`, every
/// positive trial step increases the residual norm.  With five backtracking
/// trials from `alpha_max=1`, the last evaluated candidate is `alpha=1/16`.
/// Returning `1/32` would apply a state never checked against the residual.
#[test]
fn test_line_search_fallback_returns_last_evaluated_alpha() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let u = Array3::from_elem(dims, 1.0);
    let du = Array3::from_elem(dims, 1.0);
    let f = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let alpha = coupler
        .line_search(&u, &du, &f, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    assert_eq!(alpha, 1.0 / 16.0);
    let trial = coupler.line_search_state_scratch.as_ref().unwrap();
    assert!(trial
        .iter()
        .all(|&value| (value - (1.0 + alpha)).abs() < 1e-15));
}

/// Jacobian-vector products reuse the perturbed-state workspace.
///
/// The `Density` block has no rate term, so its residual derivative is the
/// identity operator.  The finite-difference JVP must therefore return the
/// direction vector, while the scratch buffer stores the most recent
/// `u + eps * v` candidate and keeps the same allocation across calls.
#[test]
fn test_jvp_state_workspace_reuses_buffer_and_returns_identity_derivative() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Density];
    let u_prev = Array3::zeros(dims);
    let dt = 1e-6;

    let u = Array3::from_elem(dims, 2.0);
    let v = Array3::from_elem(dims, 0.25);
    let eps = 1e-8 * (1.0 + norm(&u));
    let jv = coupler
        .jacobian_vector_product(&v, &u, &u_prev, dt, dims, &field_order)
        .unwrap();
    assert!(jv.iter().all(|&value| (value - 0.25).abs() < 1e-8));

    let state = coupler.jvp_state_scratch.as_ref().unwrap();
    let first_ptr = state.as_ptr();
    let expected_state = 2.0 + eps * 0.25;
    assert!(state
        .iter()
        .all(|&value| (value - expected_state).abs() < 1e-15));

    let u = Array3::from_elem(dims, 3.0);
    let v = Array3::from_elem(dims, 0.5);
    let eps = 1e-8 * (1.0 + norm(&u));
    let jv = coupler
        .jacobian_vector_product(&v, &u, &u_prev, dt, dims, &field_order)
        .unwrap();
    assert!(jv.iter().all(|&value| (value - 0.5).abs() < 1e-8));

    let state = coupler.jvp_state_scratch.as_ref().unwrap();
    let expected_state = 3.0 + eps * 0.5;
    assert_eq!(state.as_ptr(), first_ptr);
    assert!(state
        .iter()
        .all(|&value| (value - expected_state).abs() < 1e-15));
}
