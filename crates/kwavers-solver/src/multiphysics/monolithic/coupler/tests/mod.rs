use super::super::config::NewtonKrylovConfig;
use super::super::state_vector::sorted_field_keys;
use super::*;
use crate::integration::nonlinear::GMRESConfig;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_field::UnifiedFieldType;
use kwavers_grid::Grid;
use ndarray::Array3;
use std::collections::HashMap;

mod line_search;

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
