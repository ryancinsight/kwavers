use super::*;
use kwavers_core::constants::fundamental::ACOUSTIC_IMPEDANCE_WATER_NOMINAL;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

#[test]
fn test_impedance_boundary() {
    let boundary = ImpedanceBoundary::new(ACOUSTIC_IMPEDANCE_WATER_NOMINAL, BoundaryDirections::all());

    // Test reflection coefficient
    let r = boundary.reflection_coefficient(MHZ_TO_HZ, ACOUSTIC_IMPEDANCE_WATER_NOMINAL); // Matched impedance
    assert!(r.abs() < 1e-10); // Perfect match, no reflection

    let r = boundary.reflection_coefficient(MHZ_TO_HZ, 3.0e6); // Mismatched
    assert!(r.abs() > 0.0); // Some reflection
}

#[test]
fn test_impedance_boundary_gaussian_profile() {
    let boundary = ImpedanceBoundary::new(ACOUSTIC_IMPEDANCE_WATER_NOMINAL, BoundaryDirections::all())
        .with_gaussian_profile(MHZ_TO_HZ, 0.5 * MHZ_TO_HZ);

    let medium_impedance = ACOUSTIC_IMPEDANCE_WATER_NOMINAL;

    // At center frequency, should have maximum effect
    let z_ratio_center = boundary.impedance_ratio(MHZ_TO_HZ, medium_impedance);
    assert!((z_ratio_center - 1.0).abs() < 1e-10);

    // Off center, should be attenuated by Gaussian
    let z_ratio_off = boundary.impedance_ratio(0.5 * MHZ_TO_HZ, medium_impedance);
    assert!(z_ratio_off < z_ratio_center);
}

#[test]
fn test_impedance_reflection_coefficient() {
    let boundary = ImpedanceBoundary::new(2.0e6, BoundaryDirections::all());

    // Z_target = 2.0 MRayl, Z_medium = 1.0 MRayl
    // z_ratio = 2.0
    // R = (2.0 - 1.0) / (2.0 + 1.0) = 1/3 ≈ 0.333
    let r = boundary.reflection_coefficient(MHZ_TO_HZ, 1.0e6);
    assert!((r - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_impedance_matched() {
    let boundary = ImpedanceBoundary::new(ACOUSTIC_IMPEDANCE_WATER_NOMINAL, BoundaryDirections::all());

    // Matched impedances should give zero reflection
    let r = boundary.reflection_coefficient(MHZ_TO_HZ, ACOUSTIC_IMPEDANCE_WATER_NOMINAL);
    assert!(r.abs() < 1e-12);
}

#[test]
fn test_impedance_perfect_reflector() {
    let boundary = ImpedanceBoundary::new(1e12, BoundaryDirections::all());

    // Very high target impedance (rigid wall)
    // R → +1 as Z_target → ∞
    let r = boundary.reflection_coefficient(MHZ_TO_HZ, ACOUSTIC_IMPEDANCE_WATER_NOMINAL);
    assert!(r > 0.999, "Rigid wall should have R ≈ 1, got {}", r);
}

#[test]
fn test_impedance_zero_reflector() {
    let boundary = ImpedanceBoundary::new(1.0, BoundaryDirections::all());

    // Very low target impedance (pressure release)
    // R → -1 as Z_target → 0
    let r = boundary.reflection_coefficient(MHZ_TO_HZ, ACOUSTIC_IMPEDANCE_WATER_NOMINAL);
    assert!(r < -0.999, "Pressure release should have R ≈ -1, got {}", r);
}

#[test]
fn test_impedance_boundary_spatial_apply_matched_zeros_face() {
    // Matched impedance → R = 0 → boundary cells set to zero (perfect absorption).
    use kwavers_grid::{Grid, GridAdapter};
    use ndarray::Array3;

    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let mut boundary = ImpedanceBoundary::new(
        ACOUSTIC_IMPEDANCE_WATER_NOMINAL,
        BoundaryDirections::all(),
    )
    .with_medium_impedance(ACOUSTIC_IMPEDANCE_WATER_NOMINAL);

    let mut field = Array3::<f64>::from_elem((8, 8, 8), 1.0);
    boundary
        .apply_scalar_spatial(field.view_mut(), &GridAdapter::new(grid.clone()), 0, 1e-7)
        .unwrap();

    // x_min, x_max faces zeroed
    for j in 0..8 {
        for k in 0..8 {
            assert_eq!(field[[0, j, k]], 0.0);
            assert_eq!(field[[7, j, k]], 0.0);
        }
    }
    // Interior unchanged
    assert_eq!(field[[3, 3, 3]], 1.0);
}

#[test]
fn test_impedance_boundary_spatial_apply_rigid_mirrors_face() {
    // Z_target → ∞ → R = +1 → boundary cells mirror interior (rigid wall).
    use kwavers_grid::{Grid, GridAdapter};
    use ndarray::Array3;

    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let mut boundary = ImpedanceBoundary::new(1e15, BoundaryDirections::all())
        .with_medium_impedance(ACOUSTIC_IMPEDANCE_WATER_NOMINAL);

    let mut field = Array3::<f64>::zeros((8, 8, 8));
    field[[1, 4, 4]] = 3.5; // interior next to x_min face
    field[[6, 4, 4]] = -2.5; // interior next to x_max face

    boundary
        .apply_scalar_spatial(field.view_mut(), &GridAdapter::new(grid.clone()), 0, 1e-7)
        .unwrap();

    // R ≈ 1 → ghost cell value ≈ interior cell value
    assert!((field[[0, 4, 4]] - 3.5).abs() < 1e-3);
    assert!((field[[7, 4, 4]] - (-2.5)).abs() < 1e-3);
}

#[test]
fn test_impedance_boundary_spatial_respects_directions() {
    // Only x_min direction enabled → only x_min face updated.
    use kwavers_grid::{Grid, GridAdapter};
    use ndarray::Array3;

    let grid = Grid::new(6, 6, 6, 1e-3, 1e-3, 1e-3).unwrap();
    let directions = BoundaryDirections {
        x_min: true,
        x_max: false,
        y_min: false,
        y_max: false,
        z_min: false,
        z_max: false,
    };
    let mut boundary = ImpedanceBoundary::new(
        ACOUSTIC_IMPEDANCE_WATER_NOMINAL,
        directions,
    )
    .with_medium_impedance(ACOUSTIC_IMPEDANCE_WATER_NOMINAL);

    let mut field = Array3::<f64>::from_elem((6, 6, 6), 7.0);
    boundary
        .apply_scalar_spatial(field.view_mut(), &GridAdapter::new(grid.clone()), 0, 1e-7)
        .unwrap();

    // x_min face zeroed (matched → R=0)
    assert_eq!(field[[0, 3, 3]], 0.0);
    // x_max face untouched
    assert_eq!(field[[5, 3, 3]], 7.0);
    // y, z faces untouched
    assert_eq!(field[[3, 0, 3]], 7.0);
    assert_eq!(field[[3, 5, 3]], 7.0);
}
