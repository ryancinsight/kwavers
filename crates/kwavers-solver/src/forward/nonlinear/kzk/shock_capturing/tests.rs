use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;

#[test]
fn test_shock_capture_creation() {
    let config = ShockCapturingConfig::default();
    let _capture = ShockCapture::new(config);
}

#[test]
fn test_no_shock_detection_smooth_field() {
    let config = ShockCapturingConfig {
        gradient_threshold: 1e6,
        harmonic_threshold: 0.01,
        ..Default::default()
    };
    let capture = ShockCapture::new(config);

    let mut pressure = leto::Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            let x = (i as f64 - 32.0) / 10.0;
            let y = (j as f64 - 32.0) / 10.0;
            pressure[[i, j]] = 100.0 * (-0.5 * (x * x + y * y)).exp();
        }
    }

    let result = capture
        .detect_shock(&pressure, 0.001, 0.001, SOUND_SPEED_TISSUE, MHZ_TO_HZ)
        .unwrap();
    assert!(result.max_gradient < config.gradient_threshold);
}

#[test]
fn test_shock_detection_steep_gradient() {
    let config = ShockCapturingConfig {
        gradient_threshold: 1e3,
        ..Default::default()
    };
    let capture = ShockCapture::new(config);

    let mut pressure = leto::Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            pressure[[i, j]] = if j < 32 { 1000.0 } else { -1000.0 };
        }
    }

    let result = capture
        .detect_shock(&pressure, 0.001, 0.001, SOUND_SPEED_TISSUE, MHZ_TO_HZ)
        .unwrap();
    assert!(result.shock_detected);
    assert!(result.max_gradient > 0.0);
}

#[test]
fn test_artificial_viscosity_generation() {
    let config = ShockCapturingConfig::default();
    let capture = ShockCapture::new(config);

    let mut pressure = leto::Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            pressure[[i, j]] = ((i as f64) * (j as f64)).sin() * 100.0;
        }
    }

    let q_av = capture
        .artificial_viscosity(&pressure, 0.001, 0.001, 1000.0, SOUND_SPEED_TISSUE)
        .unwrap();
    assert_eq!(q_av.dim(), (64, 64));
}

#[test]
fn test_shock_filter_application() {
    let config = ShockCapturingConfig::default();
    let capture = ShockCapture::new(config);

    let mut pressure = leto::Array2::zeros((64, 64));
    for i in 0..64 {
        for j in 0..64 {
            pressure[[i, j]] = if j < 32 { 1000.0 } else { -1000.0 };
        }
    }

    let result = capture
        .detect_shock(&pressure, 0.001, 0.001, SOUND_SPEED_TISSUE, MHZ_TO_HZ)
        .unwrap();

    capture.shock_filter(&mut pressure, &result, 3).unwrap();
}

#[test]
fn test_history_recording() {
    let config = ShockCapturingConfig::default();
    let mut capture = ShockCapture::new(config);

    let result = ShockDetectionResult {
        shock_detected: true,
        shock_location: Some(32),
        max_gradient: 5000.0,
        ..Default::default()
    };

    capture.record_result(result.clone());
    assert_eq!(capture.history().len(), 1);
    assert!(capture.history()[0].shock_detected);

    capture.clear_history();
    assert_eq!(capture.history().len(), 0);
}

#[test]
fn test_config_validation() {
    let config = ShockCapturingConfig::default();
    assert!(config.gradient_threshold > 0.0);
    assert!(config.viscosity_coefficient >= 0.0);
    assert!(config.viscosity_coefficient <= 1.0);
}

/// A pure sine at the fundamental has zero energy at all harmonics.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_harmonic_ratios_pure_sine_is_zero() {
    let config = ShockCapturingConfig {
        num_harmonics: 3,
        ..Default::default()
    };
    let capture = ShockCapture::new(config);

    let nz = 256;
    let nx = 5;
    let dz = 1.0 / 1000.0;
    let f0 = 100.0_f64;
    let mut pressure = leto::Array2::zeros((nx, nz));
    for z in 0..nz {
        let val = (TWO_PI * f0 * z as f64 * dz).sin() * 1000.0;
        for x in 0..nx {
            pressure[[x, z]] = val;
        }
    }

    let ratios = capture.compute_harmonic_ratios(&pressure, f0, dz).unwrap();

    for (n, &r) in ratios.iter().enumerate() {
        assert!(
            r < 0.01,
            "harmonic {} ratio should be ~0 for pure sine, got {:.4e}",
            n + 2,
            r
        );
    }
}

/// Synthesised signal with known second harmonic amplitude yields correct ratio.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_harmonic_ratios_known_second_harmonic() {
    let config = ShockCapturingConfig {
        num_harmonics: 3,
        ..Default::default()
    };
    let capture = ShockCapture::new(config);

    let nz = 512;
    let nx = 5;
    let dz = 1.0 / 5120.0;
    let f0 = 200.0_f64;
    let a1 = 1000.0_f64;
    let a2 = 200.0_f64;

    let mut pressure = leto::Array2::zeros((nx, nz));
    for z in 0..nz {
        let t = z as f64 * dz;
        let val = a1 * (TWO_PI * f0 * t).sin() + a2 * (TWO_PI * 2.0 * f0 * t).sin();
        for x in 0..nx {
            pressure[[x, z]] = val;
        }
    }

    let ratios = capture.compute_harmonic_ratios(&pressure, f0, dz).unwrap();

    assert!(!ratios.is_empty(), "should produce at least one ratio");
    let expected_r2 = a2 / a1;
    let measured_r2 = ratios[0];
    let err = (measured_r2 - expected_r2).abs();
    assert!(
        err < 0.02,
        "second harmonic ratio: expected {:.3}, got {:.3} (err={:.2e})",
        expected_r2,
        measured_r2,
        err
    );
}

/// Zero pressure field returns empty ratios.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_harmonic_ratios_zero_field_returns_empty() {
    let config = ShockCapturingConfig::default();
    let capture = ShockCapture::new(config);

    let pressure = leto::Array2::zeros((8, 128));
    let ratios = capture
        .compute_harmonic_ratios(&pressure, 100.0, 1e-4)
        .unwrap();
    assert!(ratios.is_empty(), "zero field should yield empty ratios");
}

/// Invalid frequency (zero) returns empty ratios without error.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_harmonic_ratios_invalid_frequency_returns_empty() {
    let config = ShockCapturingConfig::default();
    let capture = ShockCapture::new(config);

    let mut pressure = leto::Array2::zeros((8, 64));
    pressure[[4, 10]] = 1.0;
    let ratios = capture
        .compute_harmonic_ratios(&pressure, 0.0, 1e-4)
        .unwrap();
    assert!(ratios.is_empty());
}
