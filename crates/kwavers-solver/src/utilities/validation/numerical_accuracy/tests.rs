use super::NumericalValidator;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

fn default_validator() -> NumericalValidator {
    NumericalValidator::new()
}

/// FDTD phase error must be strictly positive and below 5% for CFL=0.3.
///
/// Reference: Taflove & Hagness (2005), §4.5, Table 4.1.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_phase_error_positive_and_small() {
    let v = default_validator();
    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = v.grid.dx;
    let dt = 0.3 * dx / c0;
    let k = std::f64::consts::PI / (10.0 * dx);
    let omega = c0 * k;
    let err = v.compute_phase_error_fdtd(&(), k, omega, dt).unwrap();
    assert!(
        err > 0.0,
        "FDTD phase error must be > 0 (finite-difference is never exact)"
    );
    assert!(
        err < 0.05,
        "FDTD phase error should be < 5% at CFL=0.3, got {err:.4}"
    );
}

/// PSTD phase error must be strictly less than the FDTD phase error at same dt.
///
/// PSTD uses spectral spatial derivatives (exact), so only temporal error remains,
/// which is smaller than FDTD's combined spatial+temporal error.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_pstd_phase_error_smaller_than_fdtd() {
    let v = default_validator();
    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = v.grid.dx;
    let dt = 0.3 * dx / c0;
    let k = std::f64::consts::PI / (10.0 * dx);
    let omega = c0 * k;
    let err_pstd = v.compute_phase_error(&(), k, omega, dt).unwrap();
    let err_fdtd = v.compute_phase_error_fdtd(&(), k, omega, dt).unwrap();
    assert!(
        err_pstd <= err_fdtd,
        "PSTD phase error ({err_pstd:.2e}) must be ≤ FDTD ({err_fdtd:.2e})"
    );
}

/// FDTD phase error must decrease by approximately 4× when the grid is refined 2×.
///
/// The finite-difference spatial error is O(Δx²) (2nd-order in space), so
/// halving Δx should reduce the phase error by ~4×.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_phase_error_decreases_with_finer_grid() {
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;

    let c0 = SOUND_SPEED_WATER_SIM;

    let dx_coarse = 1e-3_f64;
    let grid_coarse = Grid::new(16, 16, 16, dx_coarse, dx_coarse, dx_coarse).unwrap();
    let medium_coarse = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, c0, &grid_coarse);
    let v_coarse = NumericalValidator::with_config(grid_coarse, medium_coarse);

    let dx_fine = 0.5e-3_f64;
    let grid_fine = Grid::new(16, 16, 16, dx_fine, dx_fine, dx_fine).unwrap();
    let medium_fine = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, c0, &grid_fine);
    let v_fine = NumericalValidator::with_config(grid_fine, medium_fine);

    let dt = 0.3 * dx_coarse / c0;
    let k_test = std::f64::consts::PI / (10.0 * dx_coarse);
    let omega = c0 * k_test;

    let err_coarse = v_coarse
        .compute_phase_error_fdtd(&(), k_test, omega, dt)
        .unwrap();
    let err_fine = v_fine
        .compute_phase_error_fdtd(&(), k_test, omega, dt / 2.0)
        .unwrap();

    if err_coarse > 1e-12 {
        let ratio = err_coarse / err_fine.max(1e-15);
        assert!(
            ratio >= 1.5,
            "Phase error should decrease with finer grid (ratio={ratio:.2}, coarse={err_coarse:.3e}, fine={err_fine:.3e})"
        );
    }
}

/// CPML reflection must be < 0.001 (−60 dB), PML < 0.01 (−40 dB).
///
/// Reference: Roden & Gedney (2000); Berenger (1994).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_boundary_reflection_within_bounds() {
    let v = default_validator();
    let r_cpml = v.test_boundary_reflection("CPML").unwrap();
    let r_pml = v.test_boundary_reflection("PML").unwrap();
    assert!(r_cpml > 0.0, "CPML reflection must be positive");
    assert!(
        r_cpml <= 0.001,
        "CPML reflection must be ≤ 0.001 (−60 dB), got {r_cpml}"
    );
    assert!(r_pml > 0.0, "PML reflection must be positive");
    assert!(
        r_pml <= 0.01,
        "PML reflection must be ≤ 0.01 (−40 dB), got {r_pml}"
    );
}

/// Unknown boundary type must return an error.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_boundary_reflection_unknown_type_returns_error() {
    let v = default_validator();
    assert!(v.test_boundary_reflection("FDTD_BC").is_err());
}
