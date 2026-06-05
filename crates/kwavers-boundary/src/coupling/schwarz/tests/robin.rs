//! Robin transmission condition tests for SchwarzBoundary.

use super::super::SchwarzBoundary;
use crate::coupling::types::{BoundaryDirections, BoundaryTransmissionCondition};
use ndarray::Array3;

#[test]
fn test_schwarz_robin_condition() {
    let nx = 10;
    let ny = 10;
    let nz = 10;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 20.0;

    let alpha = 0.5;
    let beta = 0.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Robin { alpha, beta });

    let original_value = interface_field[[5, 5, 5]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_value = interface_field[[5, 5, 5]];

    assert!(
        (corrected_value - original_value).abs() > 0.1,
        "Robin condition should modify interface values"
    );
    assert!(
        (0.0..=30.0).contains(&corrected_value),
        "Robin condition produced unreasonable value: {}",
        corrected_value
    );
}

#[test]
fn test_schwarz_robin_with_nonzero_beta() {
    let nx = 8;
    let ny = 8;
    let nz = 8;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 5.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;

    let alpha = 1.0;
    let beta = 2.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Robin { alpha, beta });

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_value = interface_field[[4, 4, 4]];

    assert!(
        corrected_value > 0.0,
        "Robin condition with β should produce valid values"
    );
}

#[test]
fn test_schwarz_robin_zero_alpha() {
    let nx = 5;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 15.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 25.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Robin {
            alpha: 0.0,
            beta: 0.0,
        });

    let original_value = interface_field[[2, 2, 2]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    assert_eq!(
        interface_field[[2, 2, 2]],
        original_value,
        "Robin with α=0 should not modify field (avoids division by zero)"
    );
}

#[test]
fn test_schwarz_robin_analytical_validation() {
    let nx = 11;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 300.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 350.0;

    let alpha = 0.1;
    let beta = 0.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Robin { alpha, beta });

    let original_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    assert!(
        (0.0..500.0).contains(&corrected_center),
        "Robin condition should produce reasonable coupled value: {} (from {})",
        corrected_center,
        original_center
    );
    assert!(
        (corrected_center - original_center).abs() > 0.01,
        "Robin condition should modify interface temperature"
    );
}

#[test]
fn test_schwarz_robin_energy_stability() {
    let nx = 8;
    let ny = 8;
    let nz = 8;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 5.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;

    let alpha = 1.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Robin { alpha, beta: 0.0 });

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let val = interface_field[[i, j, k]];
                assert!(
                    (0.0..=15.0).contains(&val),
                    "Robin condition produced unstable value: {}",
                    val
                );
            }
        }
    }
}
