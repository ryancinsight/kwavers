//! Laplacian stencil accuracy and spatial-order validation tests.

use crate::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;

fn assert_quadratic_laplacian_exact(spatial_order: usize, radius: usize) {
    let grid = Grid::new(12, 12, 12, 0.2, 0.3, 0.4).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig {
        spatial_order,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                solver.pressure[[i, j, k]] = x * x + y * y + z * z;
            }
        }
    }

    solver.laplacian.fill(29.0);
    solver.calculate_laplacian(&grid).unwrap();

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let interior = (radius..grid.nx - radius).contains(&i)
                    && (radius..grid.ny - radius).contains(&j)
                    && (radius..grid.nz - radius).contains(&k);
                let expected = if interior { 6.0 } else { 29.0 };
                let actual = solver.laplacian[[i, j, k]];
                assert!(
                    (actual - expected).abs() < 1.0e-10,
                    "order {spatial_order}: laplacian[{i},{j},{k}] = {actual}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn test_westervelt_fdtd_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig::default();
    let solver = WesterveltFdtd::new(config, &grid, &medium);

    assert_eq!(solver.pressure.shape(), [32, 32, 32]);
}

#[test]
fn westervelt_laplacian_stencils_are_exact_for_quadratic_fields() {
    // Theorem: any consistent centered second-derivative stencil with
    // coefficients satisfying Σc_m=0 and Σm²c_m=2 differentiates x² exactly.
    // For p=x²+y²+z², ∇²p=2+2+2=6 on all points with complete stencil support.
    assert_quadratic_laplacian_exact(2, 1);
    assert_quadratic_laplacian_exact(4, 2);
    assert_quadratic_laplacian_exact(6, 3);
}

#[test]
fn westervelt_laplacian_rejects_unsupported_spatial_order() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig {
        spatial_order: 8,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    let err = solver.calculate_laplacian(&grid).unwrap_err();
    assert!(
        matches!(err, KwaversError::Validation(_)),
        "unsupported spatial order must return a validation error, got {err:?}"
    );
    assert_eq!(solver.config.spatial_order, 8);
}

#[test]
fn a_thin_axis_has_no_complete_three_dimensional_stencil() {
    for spatial_order in [2, 4, 6] {
        for [nx, ny, nz] in [[1, 8, 9], [8, 1, 9], [8, 9, 1]] {
            let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3).unwrap();
            let medium = HomogeneousMedium::water(&grid);
            let mut solver = WesterveltFdtd::new(
                WesterveltFdtdConfig {
                    spatial_order,
                    ..Default::default()
                },
                &grid,
                &medium,
            );
            solver.laplacian.fill(17.0);
            solver.calculate_laplacian(&grid).unwrap();
            assert!(solver.laplacian.iter().all(|value| *value == 17.0));
        }
    }
}
