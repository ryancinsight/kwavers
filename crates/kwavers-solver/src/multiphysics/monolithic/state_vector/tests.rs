use super::*;
use kwavers_field::UnifiedFieldType;
use leto::Array3;
use std::collections::HashMap;

#[test]
fn test_flatten_unflatten_round_trip() {
    let nx = 4;
    let ny = 3;
    let nz = 2;
    let mut fields = HashMap::new();
    let mut pressure = Array3::zeros((nx, ny, nz));
    pressure[[1, 1, 1]] = 42.0;
    let mut temp = Array3::zeros((nx, ny, nz));
    temp[[2, 0, 0]] = 7.0;

    fields.insert(UnifiedFieldType::Pressure, pressure);
    fields.insert(UnifiedFieldType::Temperature, temp);

    let order = sorted_field_keys(&fields);
    let flat = flatten_fields(&fields, &order);
    assert_eq!(flat.shape(), (2 * nx, ny, nz));

    let mut out_fields = HashMap::new();
    out_fields.insert(UnifiedFieldType::Pressure, Array3::zeros((nx, ny, nz)));
    out_fields.insert(UnifiedFieldType::Temperature, Array3::zeros((nx, ny, nz)));
    unflatten_fields(&flat, &mut out_fields, &order);

    assert!((out_fields[&UnifiedFieldType::Pressure][[1, 1, 1]] - 42.0).abs() < 1e-15);
    assert!((out_fields[&UnifiedFieldType::Temperature][[2, 0, 0]] - 7.0).abs() < 1e-15);
}

#[test]
fn test_field_block_view_borrows_stacked_storage() {
    let nx = 3;
    let ny = 2;
    let nz = 2;
    let stacked =
        Array3::from_shape_fn((2 * nx, ny, nz), |(i, j, k)| (100 * i + 10 * j + k) as f64);

    let view = field_block_view(&stacked, nx, 1);

    assert_eq!(view.shape(), [nx, ny, nz]);
    assert_eq!(view[[0, 1, 1]], stacked[[nx, 1, 1]]);
    let expected_ptr: *const f64 = &stacked[[nx, 0, 0]];
    assert_eq!(view.as_ptr(), expected_ptr);
}
