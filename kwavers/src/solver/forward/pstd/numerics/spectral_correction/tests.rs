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
