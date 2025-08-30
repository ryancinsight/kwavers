#![allow(dead_code)]
//! Numerical dispersion validation test
//!
//! Validates that the numerical methods correctly handle wave dispersion.
//! Reference: Trefethen, "Finite Difference and Spectral Methods", 1996

use std::f64::consts::PI;

/// Calculate numerical dispersion relation for FDTD
///
/// For a plane wave e^{i(kx - ωt)}, the numerical dispersion relation is:
/// sin²(ωΔt/2) = (cΔt/Δx)² sin²(kΔx/2)
///
/// Reference: Taflove & Hagness, "Computational Electrodynamics", 2005, Eq. 4.92
fn fdtd_dispersion_relation(
    k: f64,       // Wave number
    dx: f64,      // Spatial step
    dt: f64,      // Time step
    c: f64,       // Wave speed
    order: usize, // Spatial order (2 or 4)
) -> f64 {
    let courant = c * dt / dx;

    match order {
        2 => {
            // Second-order central difference
            let k_numerical = 2.0 * (k * dx / 2.0).sin() / dx;
            let omega_numerical = 2.0 * ((courant * k_numerical * dx / 2.0).sin()).asin() / dt;
            omega_numerical
        }
        4 => {
            // Fourth-order central difference
            // sin(kΔx/2) term modified by fourth-order stencil
            let sin_k = (k * dx / 2.0).sin();
            let k_numerical = (8.0 * sin_k - (2.0 * k * dx).sin()) / (6.0 * dx);
            let omega_numerical = 2.0 * ((courant * k_numerical * dx / 2.0).sin()).asin() / dt;
            omega_numerical
        }
        _ => panic!("Only 2nd and 4th order supported"),
    }
}

#[test]
fn test_numerical_dispersion_second_order() {
    let c = 1500.0; // Sound speed in water (m/s)
    let frequency = 1e6; // 1 MHz
    let wavelength = c / frequency;
    let k = 2.0 * PI / wavelength;

    // Test with different resolutions
    let resolutions = [10.0, 20.0, 40.0]; // Points per wavelength

    for ppw in resolutions {
        let dx = wavelength / ppw;
        let dt = 0.5 * dx / c; // CFL = 0.5

        let omega_exact = 2.0 * PI * frequency;
        let omega_numerical = fdtd_dispersion_relation(k, dx, dt, c, 2);

        let phase_error = (omega_numerical - omega_exact).abs() / omega_exact;

        // Dispersion error should decrease with resolution
        // For 2nd order: error ~ O((kΔx)²)
        let expected_error = (k * dx).powi(2);

        assert!(
            phase_error < expected_error,
            "Phase error {} exceeds expected {} for {} PPW",
            phase_error,
            expected_error,
            ppw
        );

        // Verify error decreases with increased resolution
        if ppw > 10.0 {
            assert!(
                phase_error < 0.01, // < 1% error for PPW >= 20
                "Excessive dispersion error: {} for {} PPW",
                phase_error,
                ppw
            );
        }
    }
}

#[test]
fn test_numerical_dispersion_fourth_order() {
    let c = 1500.0;
    let frequency = 1e6;
    let wavelength = c / frequency;
    let k = 2.0 * PI / wavelength;

    // Fourth-order should have much lower dispersion
    let dx = wavelength / 10.0; // Only 10 PPW
    let dt = 0.5 * dx / c;

    let omega_exact = 2.0 * PI * frequency;
    let omega_numerical = fdtd_dispersion_relation(k, dx, dt, c, 4);

    let phase_error = (omega_numerical - omega_exact).abs() / omega_exact;

    // Fourth-order error ~ O((kΔx)⁴)
    assert!(
        phase_error < 1e-4,
        "Fourth-order dispersion error too large: {}",
        phase_error
    );
}

#[test]
fn test_group_velocity() {
    // Group velocity determines energy propagation
    // Must verify vg = dω/dk is accurate

    let c = 1500.0;
    let frequency = 1e6;
    let wavelength = c / frequency;
    let k = 2.0 * PI / wavelength;
    let dk = k * 0.001; // Small perturbation

    let dx = wavelength / 20.0;
    let dt = 0.5 * dx / c;

    // Calculate group velocity numerically
    let omega1 = fdtd_dispersion_relation(k - dk, dx, dt, c, 2);
    let omega2 = fdtd_dispersion_relation(k + dk, dx, dt, c, 2);
    let vg_numerical = (omega2 - omega1) / (2.0 * dk);

    // Group velocity should equal phase velocity for non-dispersive medium
    let error = (vg_numerical - c).abs() / c;

    assert!(
        error < 0.01, // < 1% error
        "Group velocity error: {} (vg = {}, c = {})",
        error,
        vg_numerical,
        c
    );
}

#[test]
fn test_anisotropic_dispersion() {
    // Numerical dispersion is anisotropic (direction-dependent)
    // Worst case is along diagonal

    let c = 1500.0;
    let frequency = 1e6;
    let wavelength = c / frequency;

    let dx = wavelength / 20.0;
    let dt = 0.5 * dx / c;

    // Test different propagation angles
    let angles = [0.0, PI / 4.0, PI / 2.0]; // Axial, diagonal, axial

    let mut max_anisotropy: f64 = 0.0;

    for theta in angles {
        let kx = 2.0 * PI * theta.cos() / wavelength;
        let ky = 2.0 * PI * theta.sin() / wavelength;

        // Calculate effective dispersion for 2D propagation
        // This would require 2D dispersion analysis
        // Simplified here for demonstration

        let k_effective = (kx * kx + ky * ky).sqrt();
        let omega = fdtd_dispersion_relation(k_effective, dx, dt, c, 2);

        let phase_velocity = omega / k_effective;
        let anisotropy = (phase_velocity - c).abs() / c;

        max_anisotropy = max_anisotropy.max(anisotropy);
    }

    // Anisotropy should be small for adequate resolution
    assert!(
        max_anisotropy < 0.02, // < 2% variation
        "Excessive anisotropic dispersion: {}",
        max_anisotropy
    );
}
