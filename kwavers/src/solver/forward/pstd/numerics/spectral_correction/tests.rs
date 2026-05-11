use super::*;
use crate::domain::grid::Grid;
use std::f64::consts::PI;

#[test]
fn test_exact_dispersion_correction() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let dt = 1e-6;
    let c_ref = 1500.0;

    let config = SpectralCorrectionConfig {
        enabled: true,
        method: CorrectionMethod::ExactDispersion,
        cfl_number: 0.3,
        max_correction: 2.0,
    };

    let kappa = compute_spectral_correction(&grid, &config, dt, c_ref);

    assert!((kappa[[0, 0, 0]] - 1.0).abs() < 1e-10);

    for val in kappa.iter() {
        assert!(*val >= 0.5 && *val <= 2.0);
    }
}

#[test]
fn test_dispersion_error() {
    let dx = 1e-3;
    let dt = 1e-6;
    let c_ref = 1500.0;

    let k_low = PI / (10.0 * dx);
    let error_low = compute_dispersion_error(k_low, dx, dt, c_ref);
    assert!(error_low < 0.01);

    let k_high = PI / (2.0 * dx);
    let error_high = compute_dispersion_error(k_high, dx, dt, c_ref);
    assert!(error_high > error_low);
}

#[test]
fn test_phase_velocity_computation() {
    let dx = 1e-3;
    let dt = 1e-6;
    let c_ref = 1500.0;

    let c_dc = compute_numerical_phase_velocity(1e-12, dx, dt, c_ref);
    assert!((c_dc - c_ref).abs() / c_ref < 1e-6);
}

/// Theorem (Treeby & Cox 2010 Eq. 18):
///
///     kappa(k) = sinc(c_ref·dt·|k|/2) = sin(c_ref·dt·|k|/2) / (c_ref·dt·|k|/2)
///
/// for any non-zero |k|, with kappa(0) = 1 by L'Hôpital.
///
/// This test verifies the closed-form analytical agreement at five
/// representative wavenumbers spanning DC → near-Nyquist. It pins the
/// regression that surfaced as ~30% peak inflation in
/// `pykwavers/examples/na_modelling_absorption_compare.py` before the
/// kappa-inversion fix.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_treeby2010_kappa_equals_sinc() {
    // Use a 1-D grid (NY = NZ = 1) so kappa values are determined by k_x alone.
    // dx_y, dx_z are arbitrary because k_y = k_z = 0 for nz/ny = 1.
    let grid = Grid::new(64, 1, 1, 1e-4, 1e-4, 1e-4).unwrap();
    let dt = 5e-9;
    let c_ref = 1500.0;

    let config = SpectralCorrectionConfig {
        enabled: true,
        method: CorrectionMethod::Treeby2010,
        cfl_number: 0.3,
        max_correction: 10.0, // wide enough to not clip the sinc values
    };
    let kappa = compute_spectral_correction(&grid, &config, dt, c_ref);

    // DC sample: kappa(0,0,0) must be exactly 1 (sinc limit at zero).
    assert!(
        (kappa[[0, 0, 0]] - 1.0).abs() < 1e-12,
        "kappa[DC] = {} (expected 1.0)",
        kappa[[0, 0, 0]]
    );

    // Helper: expected kappa at index i for the 1-D x-axis only
    // (k_y = k_z = 0). compute_wavenumber_component uses standard FFT
    // ordering, so index 0 = DC, indices 1..N/2 = positive k, N/2 ..N-1 = negative k.
    let dx = grid.dx;
    let nx = grid.nx as i64;
    for &i in &[0_usize, 1, 4, 16, (nx as usize - 1)] {
        let kx = if (i as i64) <= nx / 2 {
            2.0 * PI * (i as f64) / (nx as f64 * dx)
        } else {
            2.0 * PI * ((i as f64) - nx as f64) / (nx as f64 * dx)
        };
        let k_mag = kx.abs();
        let arg = 0.5 * c_ref * dt * k_mag;
        let expected = if arg.abs() < 1e-12 {
            1.0
        } else {
            arg.sin() / arg
        };
        let observed = kappa[[i, 0, 0]];
        assert!(
            (observed - expected).abs() < 1e-9,
            "kappa[{}, 0, 0] = {:.10} but expected sinc(c·dt·|k|/2) = {:.10} \
             (arg = {:.6e}, k = {:.4e})",
            i,
            observed,
            expected,
            arg,
            k_mag,
        );
    }

    // Verify kappa is monotonically non-increasing as |k| grows from DC
    // (sinc is monotone on [0, π]). Up to k ≈ π/dx ≈ Nyquist, the
    // c·dt·|k|/2 argument grows but stays bounded under CFL.
    for i in 1..(grid.nx / 2) {
        let prev = kappa[[i - 1, 0, 0]];
        let curr = kappa[[i, 0, 0]];
        // Allow a tiny epsilon for floating-point noise.
        assert!(
            curr <= prev + 1e-12,
            "kappa monotonicity violated at i={}: kappa[{}]={} > kappa[{}]={}",
            i,
            i,
            curr,
            i - 1,
            prev,
        );
    }
}

#[test]
fn test_correction_methods_consistency() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let dt = 1e-6;
    let c_ref = 1500.0;

    let methods = vec![
        CorrectionMethod::ExactDispersion,
        CorrectionMethod::Treeby2010,
        CorrectionMethod::LiuPSTD,
        CorrectionMethod::SincSpatial,
    ];

    for method in methods {
        let config = SpectralCorrectionConfig {
            enabled: true,
            method,
            cfl_number: 0.3,
            max_correction: 2.0,
        };

        let kappa = compute_spectral_correction(&grid, &config, dt, c_ref);

        assert!((kappa[[0, 0, 0]] - 1.0).abs() < 0.01);

        for val in kappa.iter() {
            assert!(*val > 0.0);
        }
    }
}
