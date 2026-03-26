use super::container::{field_indices, PhysicsState};
use crate::domain::grid::Grid;
use ndarray::Array3;

#[test]
fn test_physics_state_creation() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let state = PhysicsState::new(grid);

    let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
    assert_eq!(pressure.shape(), &[10, 10, 10]);

    let mut state = state;
    state
        .initialize_field(field_indices::TEMPERATURE_IDX, 293.15)
        .unwrap();
    let temp = state.get_field(field_indices::TEMPERATURE_IDX).unwrap();

    assert!(
        (temp[[5, 5, 5]] - 293.15).abs() < f64::EPSILON,
        "Temperature initialization failed: expected 293.15, got {}",
        temp[[5, 5, 5]]
    );
}

#[test]
fn test_field_updates() {
    let grid = Grid::new(5, 5, 5, 0.1, 0.1, 0.1).unwrap();
    let mut state = PhysicsState::new(grid);

    let mut test_data = Array3::zeros((5, 5, 5));
    test_data[[2, 2, 2]] = 100.0;

    state
        .update_field(field_indices::PRESSURE_IDX, &test_data)
        .unwrap();

    let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
    assert_eq!(pressure[[2, 2, 2]], 100.0);
}

#[test]
fn test_zero_copy_access() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let mut state = PhysicsState::new(grid);

    state
        .with_field(field_indices::PRESSURE_IDX, |field| {
            assert_eq!(field.shape(), &[10, 10, 10]);
        })
        .unwrap();

    state
        .with_field_mut(field_indices::TEMPERATURE_IDX, |mut field| {
            field[[5, 5, 5]] = 300.0;
        })
        .unwrap();

    state
        .with_field(field_indices::TEMPERATURE_IDX, |field| {
            assert_eq!(field[[5, 5, 5]], 300.0);
        })
        .unwrap();
}

#[test]
fn test_field_guard_deref() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let mut state = PhysicsState::new(grid);

    state
        .initialize_field(field_indices::PRESSURE_IDX, 101325.0)
        .unwrap();

    let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
    assert_eq!(pressure[[0, 0, 0]], 101325.0);

    {
        let mut temp = state.get_field_mut(field_indices::TEMPERATURE_IDX).unwrap();
        temp[[0, 0, 0]] = 273.15;
    }

    let temp = state.get_field(field_indices::TEMPERATURE_IDX).unwrap();
    assert_eq!(temp[[0, 0, 0]], 273.15);
}
