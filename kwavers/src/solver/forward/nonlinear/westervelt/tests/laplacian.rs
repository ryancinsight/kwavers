//! Laplacian stencil accuracy and spatial-order validation tests.

use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};
use crate::KwaversError;

fn assert_quadratic_laplacian_exact(spatial_order: usize, radius: usize) {
    let grid = Grid::new(12, 12, 12, 0.2, 0.3, 0.4).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
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

    solver.calculate_laplacian(&grid).unwrap();

    for i in radius..grid.nx - radius {
        for j in radius..grid.ny - radius {
            for k in radius..grid.nz - radius {
                let actual = solver.laplacian[[i, j, k]];
                assert!(
                    (actual - 6.0).abs() < 1.0e-10,
                    "order {spatial_order}: laplacian[{i},{j},{k}] = {actual}, expected 6"
                );
            }
        }
    }
}

#[test]
fn test_westervelt_fdtd_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig::default();
    let solver = WesterveltFdtd::new(config, &grid, &medium);

    assert_eq!(solver.pressure.shape(), &[32, 32, 32]);
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
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
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
