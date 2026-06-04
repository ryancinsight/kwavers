//! Value-semantic regression tests for cylindrical projection.

use super::CylindricalMediumProjection;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::{CylindricalTopology, Grid};
use crate::heterogeneous::HeterogeneousMedium;
use crate::{CoreMedium, HomogeneousMedium};

#[test]
fn test_homogeneous_projection() {
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // Check dimensions
    assert_eq!(projection.dimensions(), (64, 32));

    // Check homogeneity
    assert!(projection.is_homogeneous());

    // Check all values are constant (homogeneous)
    let c0 = projection.sound_speed_at(0, 0);
    let rho0 = projection.density_at(0, 0);

    for iz in 0..64 {
        for ir in 0..32 {
            assert_eq!(projection.sound_speed_at(iz, ir), c0);
            assert_eq!(projection.density_at(iz, ir), rho0);
        }
    }

    // Check min/max match constant value
    assert_eq!(projection.max_sound_speed(), c0);
    assert_eq!(projection.min_sound_speed(), c0);
}

#[test]
fn test_projection_validates() {
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // Validation should pass
    projection.validate_projection().unwrap();
}

#[test]
fn test_projection_dimensions_match() {
    let grid = Grid::new(128, 128, 128, 0.00005, 0.00005, 0.00005).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(128, 64, 0.00005, 0.00005).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    let c_field = projection.sound_speed_field();
    assert_eq!(c_field.shape(), &[128, 64]);

    let rho_field = projection.density_field();
    assert_eq!(rho_field.shape(), &[128, 64]);
}

#[test]
fn test_projection_physical_bounds() {
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // All sound speeds should be positive
    for &c in projection.sound_speed_field().iter() {
        assert!(c > 0.0);
        assert!(c.is_finite());
    }

    // All densities should be positive
    for &rho in projection.density_field().iter() {
        assert!(rho > 0.0);
        assert!(rho.is_finite());
    }

    // All absorptions should be non-negative
    for &alpha in projection.absorption_field().iter() {
        assert!(alpha >= 0.0);
        assert!(alpha.is_finite());
    }
}

#[test]
fn test_accessor_methods() {
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // Test accessors return same values as array indexing
    let c_from_accessor = projection.sound_speed_at(10, 15);
    let c_from_field = projection.sound_speed_field()[[10, 15]];
    assert_eq!(c_from_accessor, c_from_field);

    let rho_from_accessor = projection.density_at(20, 10);
    let rho_from_field = projection.density_field()[[20, 10]];
    assert_eq!(rho_from_accessor, rho_from_field);
}

#[test]
fn test_spacing_and_dimensions() {
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.00015).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    assert_eq!(projection.dimensions(), (64, 32));
    assert_eq!(projection.spacing(), (0.0001, 0.00015));
}

// Property-based tests for mathematical invariants
#[test]
fn test_property_homogeneity_preservation() {
    // Property: Homogeneous 3D medium → Uniform 2D field
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // All projected values should be identical
    let c0 = projection.sound_speed_at(0, 0);
    let rho0 = projection.density_at(0, 0);
    let alpha0 = projection.absorption_at(0, 0);

    for iz in 0..64 {
        for ir in 0..32 {
            assert_eq!(
                projection.sound_speed_at(iz, ir),
                c0,
                "Sound speed must be uniform for homogeneous medium"
            );
            assert_eq!(
                projection.density_at(iz, ir),
                rho0,
                "Density must be uniform for homogeneous medium"
            );
            assert_eq!(
                projection.absorption_at(iz, ir),
                alpha0,
                "Absorption must be uniform for homogeneous medium"
            );
        }
    }
}

#[test]
fn test_property_sound_speed_bounds() {
    // Property: min(c_3D) ≤ min(c_2D) ≤ max(c_2D) ≤ max(c_3D)
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    let c_3d_max = medium.max_sound_speed();
    let c_2d_max = projection.max_sound_speed();
    let c_2d_min = projection.min_sound_speed();

    // Check bounds
    assert!(
        c_2d_min <= c_2d_max,
        "Min sound speed must be <= max sound speed"
    );
    assert!(
        c_2d_max <= c_3d_max,
        "Projected max must not exceed 3D max: {} > {}",
        c_2d_max,
        c_3d_max
    );
}

#[test]
fn test_property_positive_density() {
    // Property: ∀(iz,ir): ρ(iz,ir) > 0
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    for iz in 0..64 {
        for ir in 0..32 {
            let rho = projection.density_at(iz, ir);
            assert!(rho > 0.0, "Density must be positive at ({}, {})", iz, ir);
            assert!(
                rho.is_finite(),
                "Density must be finite at ({}, {})",
                iz,
                ir
            );
        }
    }
}

#[test]
fn test_property_non_negative_absorption() {
    // Property: ∀(iz,ir): α(iz,ir) ≥ 0
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    for iz in 0..64 {
        for ir in 0..32 {
            let alpha = projection.absorption_at(iz, ir);
            assert!(
                alpha >= 0.0,
                "Absorption must be non-negative at ({}, {})",
                iz,
                ir
            );
            assert!(
                alpha.is_finite(),
                "Absorption must be finite at ({}, {})",
                iz,
                ir
            );
        }
    }
}

#[test]
fn test_property_array_dimensions() {
    // Property: Projected arrays have shape (nz, nr)
    let grid = Grid::new(128, 128, 128, 0.00005, 0.00005, 0.00005).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(96, 48, 0.00005, 0.00005).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    assert_eq!(projection.sound_speed_field().shape(), &[96, 48]);
    assert_eq!(projection.density_field().shape(), &[96, 48]);
    assert_eq!(projection.absorption_field().shape(), &[96, 48]);
    assert_eq!(projection.dimensions(), (96, 48));
}

#[test]
fn test_heterogeneous_projection() {
    // Test projection of heterogeneous medium
    let grid = Grid::new(32, 32, 32, 0.0001, 0.0001, 0.0001).unwrap();
    let mut medium = HeterogeneousMedium::new(32, 32, 32, false);

    // Create gradient in sound speed
    for i in 0..32 {
        for j in 0..32 {
            for k in 0..32 {
                let c = SOUND_SPEED_WATER_SIM + (i as f64) * 10.0; // Gradient along x/r direction
                medium.sound_speed[[i, j, k]] = c;
                medium.density[[i, j, k]] = DENSITY_WATER_NOMINAL;
            }
        }
    }

    let topology = CylindricalTopology::new(32, 16, 0.0001, 0.0001).unwrap();
    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // Check that projection is not homogeneous
    assert!(!projection.is_homogeneous());

    // Check that min/max are correctly computed
    assert!(projection.min_sound_speed() >= SOUND_SPEED_WATER_SIM);
    assert!(projection.max_sound_speed() > projection.min_sound_speed());

    // Validate physical bounds
    projection.validate_projection().unwrap();
}

#[test]
fn test_projection_with_nonlinearity() {
    // Test that nonlinearity is correctly projected
    let grid = Grid::new(32, 32, 32, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid); // Water has B/A nonlinearity
    let topology = CylindricalTopology::new(32, 16, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // Water should have nonlinearity
    if let Some(nonlin_field) = projection.nonlinearity_field() {
        assert_eq!(nonlin_field.shape(), &[32, 16]);

        // Check all values are in physical range for B/A (typically 1-20)
        for &b_over_a in nonlin_field.iter() {
            assert!(
                (0.0..=100.0).contains(&b_over_a),
                "B/A out of physical range"
            );
        }
    }
}

#[test]
fn test_projection_index_consistency() {
    // Property: Accessor methods consistent with array views
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

    let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

    // Check several random points
    for iz in [0, 10, 30, 50, 63].iter().copied() {
        for ir in [0, 5, 15, 25, 31].iter().copied() {
            assert_eq!(
                projection.sound_speed_at(iz, ir),
                projection.sound_speed_field()[[iz, ir]]
            );
            assert_eq!(
                projection.density_at(iz, ir),
                projection.density_field()[[iz, ir]]
            );
            assert_eq!(
                projection.absorption_at(iz, ir),
                projection.absorption_field()[[iz, ir]]
            );
        }
    }
}

#[test]
fn test_projection_validates_bounds() {
    // Test that out-of-bounds topology fails gracefully
    let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
    let medium = HomogeneousMedium::water(&grid);

    // Create topology that exceeds grid bounds
    let topology_too_large = CylindricalTopology::new(128, 64, 0.0001, 0.0001).unwrap();

    // Should fail because z_max exceeds grid bounds
    let result = CylindricalMediumProjection::new(&medium, &grid, &topology_too_large);
    assert!(result.is_err(), "Should fail for out-of-bounds topology");
}
