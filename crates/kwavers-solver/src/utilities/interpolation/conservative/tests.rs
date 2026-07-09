use super::interpolator::UtilConservativeInterpolator;
use super::mode::ConservationMode;
use approx::assert_relative_eq;
use kwavers_grid::Grid;
use leto::Array3;

/// Transfer from a grid to itself must be the identity.
/// Conservation error must be < 1e-12.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_conservative_interpolator_same_grid() {
    let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1).unwrap();
    let interpolator =
        UtilConservativeInterpolator::new(&grid, &grid, ConservationMode::Energy).unwrap();

    let mut source = Array3::zeros((16, 16, 16));
    source[[8, 8, 8]] = 1.0;

    let mut target = Array3::zeros((16, 16, 16));
    interpolator.transfer(&source, &mut target).unwrap();

    assert_relative_eq!(target[[8, 8, 8]], 1.0, epsilon = 1e-12);
    assert!(
        interpolator.verify_conservation(&source, &target) < 1e-12,
        "Conservation error too large"
    );
}

/// 2:1 coarsening of a uniform field must yield a uniform target.
/// Conservation error < 1e-10.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_conservative_interpolator_coarsening() {
    let source_grid = Grid::new(32, 32, 32, 0.1, 0.1, 0.1).unwrap();
    let target_grid = Grid::new(16, 16, 16, 0.2, 0.2, 0.2).unwrap();

    let interpolator =
        UtilConservativeInterpolator::new(&source_grid, &target_grid, ConservationMode::Mass)
            .unwrap();

    let source = Array3::from_elem((32, 32, 32), 1.0);
    let mut target = Array3::zeros((16, 16, 16));
    interpolator.transfer(&source, &mut target).unwrap();

    for &val in target.iter() {
        assert_relative_eq!(val, 1.0, epsilon = 1e-10);
    }

    let err = interpolator.verify_conservation(&source, &target);
    assert!(
        err < 1e-10,
        "Conservation error {err:.2e} exceeds tolerance"
    );
}

/// 1:2 refinement of a localized field: integral must be preserved.
/// Conservation error < 1e-10.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_conservative_interpolator_refinement() {
    let source_grid = Grid::new(8, 8, 8, 0.2, 0.2, 0.2).unwrap();
    let target_grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1).unwrap();

    let interpolator =
        UtilConservativeInterpolator::new(&source_grid, &target_grid, ConservationMode::Energy)
            .unwrap();

    let mut source = Array3::zeros((8, 8, 8));
    source[[4, 4, 4]] = 8.0;

    let mut target = Array3::zeros((16, 16, 16));
    interpolator.transfer(&source, &mut target).unwrap();

    let err = interpolator.verify_conservation(&source, &target);
    assert!(
        err < 1e-10,
        "Conservation error {err:.2e} exceeds tolerance"
    );
}

/// Polynomial field f(x,y,z) = x + 2y + 3z: integral must be conserved on 2:1 coarsening.
/// Conservation error < 1e-10 (Grandy 1999).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_conservation_polynomial_field() {
    let source_grid = Grid::new(20, 20, 20, 0.05, 0.05, 0.05).unwrap();
    let target_grid = Grid::new(10, 10, 10, 0.10, 0.10, 0.10).unwrap();

    let interpolator =
        UtilConservativeInterpolator::new(&source_grid, &target_grid, ConservationMode::All)
            .unwrap();

    let mut source = Array3::zeros((20, 20, 20));
    for iz in 0..20 {
        for iy in 0..20 {
            for ix in 0..20 {
                let x = ix as f64 * 0.05;
                let y = iy as f64 * 0.05;
                let z = iz as f64 * 0.05;
                source[[ix, iy, iz]] = x + 2.0 * y + 3.0 * z;
            }
        }
    }

    let mut target = Array3::zeros((10, 10, 10));
    interpolator.transfer(&source, &mut target).unwrap();

    let err = interpolator.verify_conservation(&source, &target);
    assert!(
        err < 1e-10,
        "Conservation error {err:.2e} exceeds tolerance"
    );
}

#[test]
fn test_conservation_mode_enum() {
    assert_eq!(ConservationMode::Mass, ConservationMode::Mass);
    assert_ne!(ConservationMode::Mass, ConservationMode::Energy);
}
