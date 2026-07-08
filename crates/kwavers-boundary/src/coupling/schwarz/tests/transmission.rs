//! Dirichlet and optimized Schwarz transmission tests.

use super::super::SchwarzBoundary;
use crate::coupling::types::{BoundaryDirections, BoundaryTransmissionCondition};
use leto::Array3;

#[test]
fn test_schwarz_dirichlet_transmission() {
    let nx = 5;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 100.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 200.0;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Dirichlet);

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    assert_eq!(
        interface_field[[2, 2, 2]],
        200.0,
        "Dirichlet transmission should copy neighbor values"
    );
}

#[test]
fn test_schwarz_optimized_relaxation() {
    // u_new = (1-θ)u_old + θ*u_neighbor
    let nx = 5;
    let ny = 5;
    let nz = 5;

    let mut interface_field = Array3::<f64>::ones((nx, ny, nz)) * 10.0;
    let neighbor_field = Array3::<f64>::ones((nx, ny, nz)) * 30.0;

    let theta = 0.7;

    let boundary = SchwarzBoundary::new(1.0, BoundaryDirections::all())
        .with_transmission_condition(BoundaryTransmissionCondition::Optimized)
        .with_relaxation(theta);

    let mut interface_view = interface_field.view_mut();
    boundary.apply_transmission(&mut interface_view, &neighbor_field);

    let expected = (1.0 - theta) * 10.0 + theta * 30.0;
    assert!(
        (interface_field[[2, 2, 2]] - expected).abs() < 1e-10,
        "Optimized Schwarz relaxation failed: got {}, expected {}",
        interface_field[[2, 2, 2]],
        expected
    );
}
