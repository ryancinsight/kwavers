//! Neumann transmission condition tests for SchwarzBoundary.

use super::super::SchwarzBoundary;
use crate::boundary::coupling::types::{BoundaryDirections, BoundaryTransmissionCondition};
use ndarray::Array3;

#[test]
fn test_schwarz_neumann_flux_continuity() {
    let nx = 10;
    let ny = 10;
    let nz = 10;

    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let mut neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64;
                interface_field[[i, j, k]] = 2.0 * x;
                neighbor_field[[i, j, k]] = 2.0 * x + 5.0;
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Neumann);

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let mid = nx / 2;
    let original_value = 2.0 * (mid as f64);
    let corrected_value = interface_field[[mid, ny / 2, nz / 2]];

    assert!(
        (corrected_value - original_value).abs() < 1.0,
        "Neumann flux correction out of expected range: {} vs {}",
        corrected_value,
        original_value
    );
}

#[test]
fn test_schwarz_neumann_gradient_matching() {
    let nx = 8;
    let ny = 8;
    let nz = 8;

    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let mut neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                interface_field[[i, j, k]] = 1.0 * (i as f64);
                neighbor_field[[i, j, k]] = 3.0 * (i as f64);
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Neumann);

    let original_mid = interface_field[[4, 4, 4]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_mid = interface_field[[4, 4, 4]];

    assert!(
        corrected_mid != original_mid,
        "Neumann condition should modify field when gradients differ"
    );
}

#[test]
fn test_schwarz_neumann_analytical_validation() {
    let nx = 21;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let mut neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64;
                interface_field[[i, j, k]] = 100.0 + 5.0 * x;
                neighbor_field[[i, j, k]] = 100.0 + 5.0 * x;
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Neumann);

    let original_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let corrected_center = interface_field[[nx / 2, ny / 2, nz / 2]];

    assert!(
        (corrected_center - original_center).abs() < 0.5,
        "Neumann flux continuity should preserve matching gradients: {} vs {}",
        corrected_center,
        original_center
    );
}

#[test]
fn test_schwarz_neumann_conservation() {
    let nx = 16;
    let ny = 16;
    let nz = 16;

    let mut interface_field = Array3::<f64>::zeros((nx, ny, nz));
    let neighbor_field = Array3::<f64>::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                interface_field[[i, j, k]] = 3.0 * (i as f64);
            }
        }
    }

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Neumann);

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let mut total_correction = 0.0;
    let mut count = 0;
    for i in 1..nx - 1 {
        for j in 0..ny {
            for k in 0..nz {
                let grad = (interface_field[[i + 1, j, k]] - interface_field[[i - 1, j, k]]) / 2.0;
                total_correction += grad.abs();
                count += 1;
            }
        }
    }
    let avg_gradient = total_correction / (count as f64);

    assert!(
        (avg_gradient - 3.0).abs() < 1.0,
        "Neumann condition should preserve gradient structure: avg_grad = {}",
        avg_gradient
    );
}
