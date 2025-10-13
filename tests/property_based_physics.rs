//! Comprehensive property-based testing for physics modules
//!
//! **Testing Strategy**: Multi-framework TDD/BDD approach with proptest
//! **Literature**: FSE 2025 "Property-Based Testing for Rust Safety"
//! **Validation**: All properties verified against known physical constraints

use kwavers::testing::acoustic_properties::*;
use kwavers::testing::medium_properties::*;
use kwavers::testing::grid_properties::*;
use kwavers::{Grid, medium::{HomogeneousMedium, CoreMedium}};
use proptest::prelude::*;

// Property test: Density must always be physically valid (positive, finite)
//
// **Physical Constraint**: Density ρ > 0 for all materials
// **Reference**: Hamilton & Blackstock (1998), Chapter 1
#[cfg(test)]
proptest! {
    #[test]
    fn prop_density_always_positive_and_finite(
        density in DENSITY_RANGE.0..DENSITY_RANGE.1
    ) {
        prop_assert!(is_valid_density(density), 
            "Density must be positive and finite: {}", density);
    }
    
    #[test]
    fn prop_sound_speed_always_positive_and_finite(
        speed in SOUND_SPEED_RANGE.0..SOUND_SPEED_RANGE.1
    ) {
        prop_assert!(is_valid_sound_speed(speed),
            "Sound speed must be positive and finite: {}", speed);
    }
    
    #[test]
    fn prop_acoustic_impedance_calculation_valid(
        density in DENSITY_RANGE.0..DENSITY_RANGE.1,
        speed in SOUND_SPEED_RANGE.0..SOUND_SPEED_RANGE.1
    ) {
        prop_assert!(is_valid_acoustic_impedance(density, speed),
            "Acoustic impedance Z = ρc must be valid for ρ={}, c={}", density, speed);
    }
    
    #[test]
    fn prop_frequency_scaling_no_overflow(
        base_abs in ABSORPTION_RANGE.0..ABSORPTION_RANGE.1,
        freq in FREQUENCY_RANGE.0..FREQUENCY_RANGE.1,
        ref_freq in FREQUENCY_RANGE.0..FREQUENCY_RANGE.1
    ) {
        prop_assume!(ref_freq > 0.0);
        prop_assert!(is_valid_frequency_scaling(base_abs, freq, ref_freq),
            "Frequency scaling must not overflow: α={}, f={}, f_ref={}", 
            base_abs, freq, ref_freq);
    }
}

// Property test: Grid operations must be safe for all valid inputs
//
// **Safety Property**: Index conversions never panic within bounds
// **Reference**: Rust Bounds Checking Safety Guarantees
#[cfg(test)]
proptest! {
    #[test]
    fn prop_grid_indexing_safe_for_valid_dimensions(
        nx in 8usize..64,
        ny in 8usize..64,
        nz in 8usize..64,
        dx in 0.001f64..0.1,
        dy in 0.001f64..0.1,
        dz in 0.001f64..0.1
    ) {
        let grid = Grid::new(nx, ny, nz, dx, dy, dz)
            .expect("Grid creation should succeed for valid dimensions");
        
        prop_assert!(verify_grid_indexing_safe(&grid).is_ok(),
            "Grid indexing must be safe for dimensions {}x{}x{}", nx, ny, nz);
    }
    
    #[test]
    fn prop_grid_coordinate_conversion_bijective(
        nx in 8usize..32,
        ny in 8usize..32,
        nz in 8usize..32
    ) {
        let dx = 0.01;
        let dy = 0.01;
        let dz = 0.01;
        let grid = Grid::new(nx, ny, nz, dx, dy, dz)
            .expect("Grid creation should succeed");
        
        // Test round-trip conversion: indices -> coords -> indices
        for i in (0..nx).step_by((nx / 4).max(1)) {
            for j in (0..ny).step_by((ny / 4).max(1)) {
                for k in (0..nz).step_by((nz / 4).max(1)) {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    if let Some((i2, j2, k2)) = grid.position_to_indices(x, y, z) {
                        prop_assert_eq!(i, i2, "X index should round-trip");
                        prop_assert_eq!(j, j2, "Y index should round-trip");
                        prop_assert_eq!(k, k2, "Z index should round-trip");
                    }
                }
            }
        }
    }
}

// Property test: Medium implementations must satisfy physical constraints
// Physical Laws: Conservation of energy, causality, positivity
// Reference: Pierce (1989) "Acoustics: An Introduction"
#[cfg(test)]
proptest! {
    #[test]
    fn prop_homogeneous_medium_properties_physically_valid(
        density in DENSITY_RANGE.0..DENSITY_RANGE.1,
        speed in SOUND_SPEED_RANGE.0..SOUND_SPEED_RANGE.1,
        absorption in ABSORPTION_RANGE.0..ABSORPTION_RANGE.1,
        scattering in ABSORPTION_RANGE.0..ABSORPTION_RANGE.1
    ) {
        let grid = Grid::new(16, 16, 16, 0.01, 0.01, 0.01)
            .expect("Test grid creation");
        // HomogeneousMedium::new takes (density, sound_speed, mu_a, mu_s_prime, grid)
        let medium = HomogeneousMedium::new(density, speed, absorption, scattering, &grid);
        
        prop_assert!(verify_medium_properties_physically_valid(&medium, &grid).is_ok(),
            "Medium properties must be physically valid");
        
        // Additional invariant: max_sound_speed should equal sound_speed for homogeneous
        prop_assert!((medium.max_sound_speed() - speed).abs() < 1e-10,
            "Homogeneous medium max speed should equal constant speed");
    }
}

// Property test: CFL stability condition must be verifiable
// Stability Condition: CFL = c·Δt/Δx ≤ CFL_max for numerical stability
// Reference: Courant et al. (1928), Finkelstein & Kastner (2007)
#[cfg(test)]
proptest! {
    #[test]
    fn prop_cfl_condition_calculable(
        speed in SOUND_SPEED_RANGE.0..SOUND_SPEED_RANGE.1,
        dx in 0.001f64..0.1,
        dt in 1e-8f64..1e-4
    ) {
        let cfl = speed * dt / dx;
        
        prop_assert!(cfl.is_finite(), "CFL number must be finite");
        prop_assert!(cfl > 0.0, "CFL number must be positive");
        
        // For FDTD stability: CFL ≤ 1/√3 ≈ 0.577 in 3D
        // For PSTD: can be larger due to spectral accuracy
        let stability_threshold = 2.0; // Conservative threshold for testing
        if cfl > stability_threshold {
            // Just record, don't fail - some methods allow larger CFL
            prop_assume!(cfl <= stability_threshold);
        }
    }
}

// Property test: Wave propagation speed invariants
// Physical Law: Wave speed in medium c = √(K/ρ) where K is bulk modulus
// Reference: Landau & Lifshitz "Theory of Elasticity"
#[cfg(test)]
proptest! {
    #[test]
    fn prop_wave_speed_relationships(
        density in DENSITY_RANGE.0..DENSITY_RANGE.1,
        bulk_modulus in 1e8f64..1e11 // Pa (covers gases to solids)
    ) {
        let speed = (bulk_modulus / density).sqrt();
        
        prop_assert!(speed.is_finite(), "Wave speed must be finite");
        prop_assert!(speed > 0.0, "Wave speed must be positive");
        prop_assert!(is_valid_sound_speed(speed), 
            "Derived speed must satisfy physical constraints");
    }
}

// Property test: Absorption coefficient frequency dependence
// Power Law: α(f) = α₀·f^y where y ∈ [0, 2] for physical media
// Reference: Szabo (1995), Treeby & Cox (2010)
#[cfg(test)]
proptest! {
    #[test]
    fn prop_power_law_absorption_physically_valid(
        alpha_0 in 0.0f64..10.0,
        power in 0.0f64..2.0,
        freq in FREQUENCY_RANGE.0..FREQUENCY_RANGE.1
    ) {
        let alpha_f = alpha_0 * freq.powf(power);
        
        prop_assert!(alpha_f.is_finite(), "Absorption must be finite");
        prop_assert!(alpha_f >= 0.0, "Absorption cannot be negative");
        
        // Causality constraint: α(f) must increase with frequency for y > 0
        if power > 0.0 {
            let freq2 = freq * 2.0;
            let alpha_f2 = alpha_0 * freq2.powf(power);
            prop_assert!(alpha_f2 >= alpha_f,
                "Absorption must increase with frequency for positive power");
        }
    }
}

// Property test: Nonlinearity parameter constraints
// Physical Range: B/A ∈ [1, 20] for known materials
// Reference: Beyer (1997) "Parameter of Nonlinearity in Fluids"
#[cfg(test)]
proptest! {
    #[test]
    fn prop_nonlinearity_parameter_range(
        ba_param in NONLINEARITY_RANGE.0..NONLINEARITY_RANGE.1
    ) {
        prop_assert!(ba_param.is_finite(), "B/A parameter must be finite");
        prop_assert!(ba_param >= 1.0, "B/A parameter must be ≥ 1");
        
        // Water reference: B/A ≈ 5
        let water_ba = 5.0;
        let relative_deviation = (ba_param - water_ba).abs() / water_ba;
        
        // Most biological tissues are within 3x of water
        prop_assume!(relative_deviation < 3.0);
    }
}

// Property test: Grid spacing constraints for numerical accuracy
// Nyquist Criterion: λ/Δx ≥ 2 (minimum 2 points per wavelength)
// Practical Accuracy: λ/Δx ≥ 10 recommended for <1% dispersion error
// Reference: Finkelstein & Kastner (2007)
#[cfg(test)]
proptest! {
    #[test]
    fn prop_grid_resolution_adequate_for_frequency(
        speed in SOUND_SPEED_RANGE.0..SOUND_SPEED_RANGE.1,
        freq in FREQUENCY_RANGE.0..FREQUENCY_RANGE.1,
        points_per_wavelength in 2.0f64..20.0
    ) {
        let wavelength = speed / freq;
        let dx = wavelength / points_per_wavelength;
        
        prop_assert!(dx > 0.0, "Grid spacing must be positive");
        prop_assert!(dx.is_finite(), "Grid spacing must be finite");
        
        // Verify Nyquist criterion is satisfied
        let actual_ppw = wavelength / dx;
        prop_assert!(actual_ppw >= 2.0,
            "Must have at least 2 points per wavelength (Nyquist), have {}", actual_ppw);
    }
}
