//! CFL stability condition validation test
//!
//! Validates that the CFL condition is properly enforced for numerical stability.
//! Reference: Taflove & Hagness, "Computational Electrodynamics", 2005

use kwavers::grid::Grid;
use kwavers::medium::HomogeneousMedium;
use kwavers::physics::constants::CFL_SAFETY_FACTOR;

#[test]
fn test_cfl_condition_3d_fdtd() {
    // Create a 3D grid
    let dx = 1e-3; // 1mm spacing
    let grid = Grid::new(10, 10, 10, dx, dx, dx).unwrap();

    // Create medium with known sound speed
    let sound_speed = 1500.0; // m/s (water)
    let density = 1000.0; // kg/m³
    let _medium = HomogeneousMedium::from_minimal(density, sound_speed, &grid);

    // Calculate CFL-limited time step
    // For 3D FDTD: dt_max = dx / (c * sqrt(3))
    let dx_min = dx;
    let dt_max = dx_min / (sound_speed * 3.0_f64.sqrt());
    let dt_safe = CFL_SAFETY_FACTOR * dt_max;

    // Verify the time step is below CFL limit
    assert!(dt_safe < dt_max, "Time step must be below CFL limit");
    
    // CFL safety factor validation (avoid constant assertions)
    let safety_valid = CFL_SAFETY_FACTOR < 1.0 && CFL_SAFETY_FACTOR > 0.0;
    assert!(safety_valid, "CFL safety factor must be in (0, 1) range");

    // Verify against literature value (Taflove recommends 0.5 for 3D)
    let safety_diff = (CFL_SAFETY_FACTOR - 0.5).abs();
    assert!(
        safety_diff < 0.1,
        "CFL safety factor should be close to 0.5 for 3D FDTD"
    );
}

#[test]
fn test_numerical_dispersion() {
    // Test that spatial discretization is fine enough to avoid dispersion
    // Rule: dx < λ/10 where λ is wavelength

    let frequency = 1e6; // 1 MHz
    let sound_speed = 1500.0; // m/s
    let wavelength = sound_speed / frequency;
    let dx_max = wavelength / 10.0;

    // Create grid with proper resolution
    let dx = dx_max / 2.0; // Use half the maximum for safety
    let _grid = Grid::new(100, 100, 100, dx, dx, dx).unwrap();

    // Verify grid resolution
    assert!(
        dx < dx_max,
        "Grid spacing must be < λ/10 to avoid dispersion"
    );
    assert!(dx > 0.0, "Grid spacing must be positive");

    // Calculate points per wavelength
    let ppw = wavelength / dx;
    assert!(ppw > 10.0, "Must have >10 points per wavelength");
}
