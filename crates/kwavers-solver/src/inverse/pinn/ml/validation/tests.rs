use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use ndarray::Array2;

#[test]
fn test_compute_validation_metrics() {
    let reference = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64);
    let prediction = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64 + 0.1);

    let metrics = compute_validation_metrics(&reference, &prediction).unwrap();

    assert!(metrics.mean_absolute_error > 0.0);
    assert!(metrics.rmse > 0.0);
    assert!(metrics.max_error >= metrics.mean_absolute_error);
}

#[test]
fn test_compute_correlation() {
    let reference = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64);
    let prediction = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64 * 1.1);

    let corr = compute_correlation(&reference, &prediction).unwrap();

    // Linear relationship → correlation = 1.0
    assert!(corr > 0.9);
}

#[test]
fn test_validation_report_passes() {
    let metrics = PinnValidationMetrics {
        mean_absolute_error: 0.01,
        rmse: 0.02,
        relative_l2_error: 0.03,
        max_error: 0.05,
    };

    let report = PinnValidationReport {
        metrics,
        correlation: 0.99,
        mean_relative_error_percent: 3.0,
        num_points: 1000,
        fdtd_time_secs: 1.0,
        pinn_time_secs: 0.001,
        speedup_factor: 1000.0,
    };

    assert!(report.passes(0.10));
    assert!(!report.passes(0.02));
}

#[test]
fn test_validation_report_summary() {
    let metrics = PinnValidationMetrics {
        mean_absolute_error: 0.01,
        rmse: 0.02,
        relative_l2_error: 0.03,
        max_error: 0.05,
    };

    let report = PinnValidationReport {
        metrics,
        correlation: 0.99,
        mean_relative_error_percent: 3.0,
        num_points: 1000,
        fdtd_time_secs: 1.0,
        pinn_time_secs: 0.001,
        speedup_factor: 1000.0,
    };

    let summary = report.summary();
    assert!(summary.contains("PINN Validation Report"));
    assert!(summary.contains("Speedup: 1000.0×"));
    assert!(summary.contains("✅ PASS"));
}

#[test]
fn test_validate_pinn_vs_fdtd() {
    use crate::inverse::pinn::ml::burn_wave_equation_1d::BurnPINN1DWave;
    use crate::inverse::pinn::ml::fdtd_reference::FDTDConfig;
    use crate::inverse::pinn::ml::BurnPINNConfig;
    use coeus_core::MoiraiBackend;

    type Backend = MoiraiBackend;
    let pinn = BurnPINN1DWave::<Backend>::new(BurnPINNConfig::default()).unwrap();

    let fdtd_config = FDTDConfig {
        wave_speed: SOUND_SPEED_WATER_SIM,
        nx: 50,
        nt: 50,
        dx: 0.01,
        dt: 0.000005,
        ..Default::default()
    };

    let report = validate_pinn_vs_fdtd(&pinn, fdtd_config).unwrap();
    assert!(report.num_points > 0);
    assert!(report.speedup_factor > 0.0);
    assert!(report.correlation.abs() <= 1.0);
}
