//! Property-based tests for PhotoacousticSimulator simulate() safety and invariants
//!
//! Validates that simulate() produces time-resolved fields without panics,
//! correct dimensionality, and finite energy across randomized small grids.

use kwavers::{
    grid::Grid,
    medium::homogeneous::HomogeneousMedium,
    simulation::modalities::photoacoustic::{PhotoacousticParameters, PhotoacousticSimulator},
};
use proptest::prelude::*;

fn make_grid(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Grid {
    Grid::new(nx, ny, nz, dx, dy, dz).expect("grid parameters valid")
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 10, .. ProptestConfig::default() })]
    #[test]
    fn simulate_is_safe_and_produces_valid_outputs(
        nx in 8usize..24,
        ny in 8usize..24,
        nz in 8usize..16,
        dx in 5e-4f64..5e-3f64,
        dy in 5e-4f64..5e-3f64,
        dz in 5e-4f64..5e-3f64,
    ) {
        // Arrange
        let grid = make_grid(nx, ny, nz, dx, dy, dz);
        let params = PhotoacousticParameters::default();
        let medium = HomogeneousMedium::water(&grid);
        let mut sim = PhotoacousticSimulator::new(grid.clone(), params, &medium).unwrap();

        // Act
        let fluence = sim.compute_fluence().unwrap();
        let initial = sim.compute_initial_pressure(&fluence).unwrap();
        let result = sim.simulate(&initial).unwrap();

        // Assert: dimensions
        assert_eq!(result.reconstructed_image.dim(), (nx, ny, nz));
        assert_eq!(result.pressure_fields.len(), result.time.len());

        // Assert: values finite and not all zeros
        let mut any_nonzero = false;
        for val in result.reconstructed_image.iter() {
            let val: f64 = *val;
            assert!(val.is_finite(), "non-finite value in reconstructed image");
            if val.abs() > 0.0 { any_nonzero = true; }
        }
        assert!(any_nonzero, "reconstructed image should contain non-zero signal");
    }
}
