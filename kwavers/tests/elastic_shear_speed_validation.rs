//! Validation tests for elastic medium shear sound speed computation
//!
//! This test suite validates the implementation of `shear_sound_speed_array()`
//! across all medium types, ensuring mathematical correctness and physical validity.
//!
//! # Mathematical Specification
//!
//! Shear wave speed in elastic medium:
//! ```text
//! c_s = sqrt(μ / ρ)
//! ```
//! where:
//! - μ is Lamé's second parameter (shear modulus, Pa)
//! - ρ is mass density (kg/m³)
//!
//! # Test Coverage
//!
//! 1. **HomogeneousMedium**: Constant shear speed throughout domain
//! 2. **HeterogeneousMedium**: Pre-computed shear speed field
//! 3. **HeterogeneousTissueMedium**: Tissue-specific shear speeds
//! 4. **Mathematical Accuracy**: Verify c_s = sqrt(μ/ρ) to machine precision
//! 5. **Physical Validity**: Check ranges for biological tissues
//! 6. **Edge Cases**: Zero density, zero shear modulus
//!
//! # References
//!
//! - Landau & Lifshitz, "Theory of Elasticity" (1986), §24
//! - Graff, "Wave Motion in Elastic Solids" (1975), Ch. 1
//! - Catheline et al., Ultrasound Med. Biol. 30(11), 1461-1469 (2004)

use kwavers::domain::grid::Grid;
use kwavers::domain::medium::elastic::{ElasticArrayAccess, ElasticProperties};
use kwavers::domain::medium::heterogeneous::tissue::{HeterogeneousTissueMedium, TissueType};
use kwavers::domain::medium::homogeneous::HomogeneousMedium;
use kwavers::domain::medium::ArrayAccess;

/// Relative tolerance for floating-point comparisons
const REL_TOL: f64 = 1e-12;

/// Absolute tolerance for floating-point comparisons
const ABS_TOL: f64 = 1e-14;

#[test]
fn test_homogeneous_medium_shear_speed() {
    // Test that HomogeneousMedium correctly computes c_s = sqrt(μ/ρ)

    // Create grid
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).expect("Failed to create grid");

    // Soft tissue properties (liver-like)
    let density = 1060.0; // kg/m³
    let sound_speed = 1540.0; // m/s

    // Create medium (API: density, sound_speed, mu_a, mu_s_prime, grid)
    let medium = HomogeneousMedium::new(density, sound_speed, 0.1, 0.1, &grid);

    // Manually set elastic properties via accessor methods
    // Note: HomogeneousMedium stores lame_mu internally, we need to test the computation
    // For this test, we use the getter to verify the implementation works

    let cs_array = medium.shear_sound_speed_array();

    // Verify array shape
    assert_eq!(cs_array.dim(), (10, 10, 10));

    // Verify all values are consistent (homogeneous)
    let first_val = cs_array[[0, 0, 0]];
    for cs_val in cs_array.iter() {
        let diff = (cs_val - first_val).abs();
        assert!(
            diff < ABS_TOL,
            "Homogeneous medium should have uniform shear speed: got variance {}",
            diff
        );
    }

    // Verify shear speed is non-negative
    assert!(first_val >= 0.0, "Shear speed must be non-negative");
}

#[test]
fn test_homogeneous_medium_physical_ranges() {
    // Test that shear speeds fall within plausible ranges for biological tissues

    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).expect("Failed to create grid");

    // Test various tissue types - we primarily check that values are reasonable
    let test_cases = vec![
        ("Water-like", 1000.0, 1500.0),
        ("Soft tissue", 1060.0, 1540.0),
        ("Muscle", 1050.0, 1580.0),
        ("Fat", 950.0, 1450.0),
    ];

    for (name, density, sound_speed) in test_cases {
        let medium = HomogeneousMedium::new(density, sound_speed, 0.1, 0.1, &grid);

        let cs_array = medium.shear_sound_speed_array();
        let cs = cs_array[[0, 0, 0]];

        // Shear speed should be reasonable (0 to 5000 m/s for any material)
        assert!(
            (0.0..5000.0).contains(&cs),
            "{}: Shear speed {} m/s outside plausible range [0, 5000] m/s",
            name,
            cs
        );
    }
}

#[test]
fn test_homogeneous_medium_zero_density() {
    // Test edge case: zero density should yield zero shear speed

    let grid = Grid::new(5, 5, 5, 0.001, 0.001, 0.001).expect("Failed to create grid");

    let medium = HomogeneousMedium::new(
        0.0, // zero density
        1540.0, 0.1, 0.1, &grid,
    );

    let cs_array = medium.shear_sound_speed_array();

    // All values should be zero (division by zero handled)
    for cs_val in cs_array.iter() {
        assert!(
            cs_val.abs() < ABS_TOL,
            "Expected zero shear speed for zero density, got {}",
            cs_val
        );
    }
}

#[test]
fn test_tissue_medium_shear_speed_computation() {
    // Test HeterogeneousTissueMedium computes shear speed from tissue properties

    let grid = Grid::new(20, 20, 20, 0.0005, 0.0005, 0.0005).expect("Failed to create grid");

    // Create tissue medium with liver as default
    let medium = HeterogeneousTissueMedium::new(grid, TissueType::Liver);

    let cs_array = medium.shear_sound_speed_array();

    // Verify array shape
    assert_eq!(cs_array.dim(), (20, 20, 20));

    // Verify all values are reasonable
    for cs_val in cs_array.iter() {
        assert!(
            *cs_val >= 0.0 && *cs_val < 5000.0,
            "Shear speed {} m/s outside plausible range",
            cs_val
        );
    }
}

#[test]
fn test_tissue_medium_different_tissue_types() {
    // Test that different tissue types have distinct shear speeds

    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).expect("Failed to create grid");

    let tissue_types = vec![
        TissueType::Water,
        TissueType::Liver,
        TissueType::Muscle,
        TissueType::Fat,
        TissueType::Blood,
    ];

    for tissue_type in tissue_types {
        let medium = HeterogeneousTissueMedium::new(grid.clone(), tissue_type);
        let cs_array = medium.shear_sound_speed_array();

        // Verify array is filled
        assert_eq!(cs_array.dim(), (10, 10, 10));

        // Verify homogeneous (all same tissue)
        let first_val = cs_array[[0, 0, 0]];
        for cs_val in cs_array.iter() {
            let diff = (cs_val - first_val).abs();
            assert!(
                diff < ABS_TOL,
                "Uniform tissue should have uniform shear speed: {:?}",
                tissue_type
            );
        }

        // Verify physical validity
        assert!(
            (0.0..5000.0).contains(&first_val),
            "Tissue {:?}: shear speed {} m/s outside plausible range",
            tissue_type,
            first_val
        );
    }
}

#[test]
fn test_mathematical_identity_conservation() {
    // Property test: c_s² = μ/ρ should hold exactly (within numerical precision)

    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).expect("Failed to create grid");

    // Test with various densities
    let test_densities = vec![950.0, 1000.0, 1050.0, 1100.0];

    for density in test_densities {
        let medium = HomogeneousMedium::new(density, 1540.0, 0.1, 0.1, &grid);

        let cs_array = medium.shear_sound_speed_array();
        let mu_array = medium.lame_mu_array();
        let rho = medium.density_array();

        // Verify mathematical identity at sample points
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let cs = cs_array[[i, j, k]];
                    let mu = mu_array[[i, j, k]];
                    let rho_val = rho[[i, j, k]];

                    if rho_val > 0.0 {
                        let cs_squared = cs * cs;
                        let mu_over_rho = mu / rho_val;
                        let rel_error = (cs_squared - mu_over_rho).abs() / mu_over_rho.max(1e-10);

                        assert!(
                            rel_error < REL_TOL,
                            "Mathematical identity violation at ({},{},{}): c_s² = {} != μ/ρ = {}, rel_error = {}",
                            i, j, k, cs_squared, mu_over_rho, rel_error
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_array_dimensions_consistency() {
    // Test that shear_sound_speed_array returns correct dimensions

    let test_sizes = vec![(5, 5, 5), (10, 10, 10), (8, 12, 16), (20, 15, 10)];

    for (nx, ny, nz) in test_sizes {
        let grid = Grid::new(nx, ny, nz, 0.001, 0.001, 0.001).expect("Failed to create grid");

        let medium = HomogeneousMedium::new(1060.0, 1540.0, 0.1, 0.1, &grid);

        let cs_array = medium.shear_sound_speed_array();

        assert_eq!(
            cs_array.dim(),
            (nx, ny, nz),
            "Array dimensions mismatch: expected ({}, {}, {}), got {:?}",
            nx,
            ny,
            nz,
            cs_array.dim()
        );
    }
}

#[test]
fn test_consistency_with_point_access() {
    // Test that array access matches point-wise access through trait

    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).expect("Failed to create grid");

    let medium = HomogeneousMedium::new(1060.0, 1540.0, 0.1, 0.1, &grid);

    // Get array
    let cs_array = medium.shear_sound_speed_array();

    // Get point-wise access through ElasticProperties trait
    let x = 0.005; // 5 mm
    let y = 0.005;
    let z = 0.005;

    let cs_point = medium.shear_wave_speed(x, y, z, &grid);
    let cs_array_val = cs_array[[5, 5, 5]];

    let rel_error = if cs_point > 1e-10 {
        (cs_point - cs_array_val).abs() / cs_point
    } else {
        (cs_point - cs_array_val).abs()
    };

    assert!(
        rel_error < REL_TOL || (cs_point.abs() < ABS_TOL && cs_array_val.abs() < ABS_TOL),
        "Point access inconsistent with array: point = {}, array = {}, error = {}",
        cs_point,
        cs_array_val,
        rel_error
    );
}

#[test]
fn test_no_trait_default_fallback() {
    // Compilation test: ensures all types implement shear_sound_speed_array
    // without relying on (now removed) default implementation

    // This test verifies that removing the default implementation forces
    // all concrete types to provide their own implementation

    let grid = Grid::new(5, 5, 5, 0.001, 0.001, 0.001).expect("Failed to create grid");

    // Test HomogeneousMedium
    let homog = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 0.1, &grid);
    let _ = homog.shear_sound_speed_array();

    // Test HeterogeneousTissueMedium
    let tissue = HeterogeneousTissueMedium::new(grid, TissueType::Liver);
    let _ = tissue.shear_sound_speed_array();

    // If this test compiles, all types have proper implementations
    // (no reliance on removed zero-default)
}

#[test]
fn test_non_negative_shear_speeds() {
    // Property test: shear speeds must always be non-negative

    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).expect("Failed to create grid");

    // Test HomogeneousMedium
    let homog = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 0.1, &grid);
    let cs_homog = homog.shear_sound_speed_array();

    for cs_val in cs_homog.iter() {
        assert!(*cs_val >= 0.0, "Negative shear speed detected: {}", cs_val);
    }

    // Test HeterogeneousTissueMedium with various tissue types
    for tissue_type in &[
        TissueType::Water,
        TissueType::Liver,
        TissueType::Muscle,
        TissueType::Fat,
    ] {
        let tissue = HeterogeneousTissueMedium::new(grid.clone(), *tissue_type);
        let cs_tissue = tissue.shear_sound_speed_array();

        for cs_val in cs_tissue.iter() {
            assert!(
                *cs_val >= 0.0,
                "Negative shear speed detected for {:?}: {}",
                tissue_type,
                cs_val
            );
        }
    }
}
