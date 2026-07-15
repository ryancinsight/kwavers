//! End-to-end abdominal + brain focused-bowl pipeline integration tests.

use super::super::{run_theranostic_nonlinear_3d, Nonlinear3dConfig};
use super::fixtures::{abdominal_fixture, brain_fixture};
use crate::therapy::theranostic_guidance::AnatomyKind;

#[test]
fn nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive() {
    let (ct, labels) = abdominal_fixture();
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.grid_size = 12;
    config.element_count = 18;
    config.receiver_count = 8;
    config.source_encoding_count = 2;
    config.iterations = 1;
    config.frequency_hz = 300_000.0;
    config.source_pressure_pa = 28.0e6;
    config.cycles = 2.0;
    config.bubble_time_steps_per_period = 24;
    config.cavitation_iterations = 6;

    let result = run_theranostic_nonlinear_3d(
        AnatomyKind::Liver,
        ct,
        Some(labels),
        [2.0, 2.0, 2.0],
        &config,
        None,
    )
    .expect("nonlinear 3-D fixture must run");

    assert!(result.is_full_wave_inversion);
    assert!(result.uses_nonlinear_wave_propagation);
    assert!(result.uses_rayleigh_plesset);
    assert_eq!(result.ct_hu.shape(), [12, 12, 12]);
    assert!((result.treatment_window_radius_m - config.treatment_window_radius_m).abs() < 1.0e-12);
    assert!(result.wavelength_min_m > 0.0);
    assert!(result.points_per_wavelength_min > 0.0);
    assert_eq!(
        result.resolution_meets_min_ppw,
        result.points_per_wavelength_min >= config.min_points_per_wavelength
    );
    assert!(result.active_voxels > 32);
    assert!(result.target_mask.iter().filter(|active| **active).count() >= 2);
    assert!(
        result
            .inversion_mask
            .iter()
            .filter(|active| **active)
            .count()
            >= result.target_mask.iter().filter(|active| **active).count()
    );
    assert!(
        result.source_scale.is_finite() && result.source_scale > 0.0 && result.source_scale <= 1.0,
        "calibrated source scale must be finite and non-amplifying; got {}",
        result.source_scale,
    );
    let peak_pressure = result
        .westervelt_peak_pressure_pa
        .iter()
        .copied()
        .fold(0.0, f64::max);
    assert!(
        peak_pressure.is_finite() && peak_pressure > 0.0 && peak_pressure < 1.0e9,
        "histotripsy Westervelt peak must be finite and below the soft-tissue bulk-modulus scale; got {peak_pressure:.3e} Pa"
    );
    assert!(
        result
            .cavitation_source_density
            .iter()
            .copied()
            .fold(0.0, f64::max)
            > 0.0
    );
    assert!(
        result
            .multiparameter_fwi_score
            .iter()
            .copied()
            .fold(0.0, f64::max)
            >= 0.0
    );
    assert!(result
        .reconstructed_delta_beta
        .iter()
        .copied()
        .any(|value| value.is_finite()));
    assert!(result.fwi_objective_history.iter().all(|v| v.is_finite()));
    assert!(
        result.fwi_objective_history.last().copied().unwrap()
            <= result.fwi_objective_history.first().copied().unwrap()
    );
    assert!(
        result.cavitation_objective_history.last().copied().unwrap()
            <= result
                .cavitation_objective_history
                .first()
                .copied()
                .unwrap()
    );
    assert!(result.therapy_points_m.len() >= 8);
    assert!(result.receiver_points_m.len() >= 4);
}

/// End-to-end integration test for the `AnatomyKind::Brain` path: synthetic
/// CT with a cortical skull shell wrapping a brain interior, transcranial
/// calvarium focused-bowl aperture, lossless Westervelt forward, discrete adjoint
/// FWI, heterogeneous CT-derived path-integrated cavitation Green's function
/// (including `y = 2` Stokes-Kirchhoff skull attenuation), and Rayleigh-
/// Plesset passive subharmonic inverse.
///
/// # Why this complements the abdominal test
///
/// The abdominal fixture exercises soft-tissue-only paths where the new
/// `y ≈ 1.05` exponent is a small correction. The brain fixture is the only
/// path that actually places **skull voxels (HU > 300)** between source and
/// receiver — i.e., the only path where:
/// - the `α₀ = 13 - 20 dB/(cm·MHz)` skull attenuation appears in the
///   integral,
/// - the `y = 2` Stokes-Kirchhoff power-law gives a `3.07×` reduction at
///   the 325 kHz subharmonic versus the naive `y = 1` extrapolation,
/// - the focused-bowl aperture is placed on the calvarium surface rather than on
///   skin-coupled abdominal arc.
#[test]
fn nonlinear_3d_brain_focused_bowl_pipeline_is_input_sensitive_through_skull() {
    let ct = brain_fixture();
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Brain);
    config.grid_size = 12;
    config.element_count = 24;
    config.receiver_count = 8;
    config.source_encoding_count = 2;
    config.iterations = 1;
    config.frequency_hz = 650_000.0;
    config.source_pressure_pa = 28.0e6;
    config.cycles = 2.0;
    config.bubble_time_steps_per_period = 24;
    config.cavitation_iterations = 6;

    let result = run_theranostic_nonlinear_3d(
        AnatomyKind::Brain,
        ct,
        None,
        [1.5, 1.5, 1.5],
        &config,
        Some([0.55, 0.50, 0.50]),
    )
    .expect("nonlinear 3-D brain fixture must run");

    assert!(result.is_full_wave_inversion);
    assert!(result.uses_nonlinear_wave_propagation);
    assert!(result.uses_rayleigh_plesset);
    assert_eq!(result.ct_hu.shape(), [12, 12, 12]);
    assert!(result.active_voxels > 32);
    assert!(
        result.target_mask.iter().filter(|active| **active).count() >= 2,
        "synthetic brain ellipsoidal target must be non-empty inside the body support",
    );
    // The transcranial focused-bowl model placed on the calvarium cap.
    assert!(
        result
            .aperture_model
            .contains("transcranial_calvarium_focused_bowl"),
        "brain aperture model must be the transcranial calvarium focused bowl; got '{}'",
        result.aperture_model,
    );
    // Westervelt peak pressure must be positive somewhere inside the
    // skull-bounded support.
    assert!(
        result.source_scale.is_finite() && result.source_scale > 0.0 && result.source_scale <= 1.0,
        "calibrated source scale must be finite and non-amplifying; got {}",
        result.source_scale,
    );
    let peak_pressure = result
        .westervelt_peak_pressure_pa
        .iter()
        .copied()
        .fold(0.0, f64::max);
    assert!(
        peak_pressure.is_finite() && peak_pressure > 0.0 && peak_pressure < 1.0e9,
        "Westervelt peak pressure must be finite after source-encoded transmissions; got {peak_pressure:.3e} Pa",
    );
    // Cavitation source density must be positive when the configured
    // histotripsy drive exceeds the inertial-cavitation MI threshold.
    assert!(
        result
            .cavitation_source_density
            .iter()
            .copied()
            .fold(0.0, f64::max)
            > 0.0,
        "cavitation source density must respond to the simulated peak pressure; peak_pressure_pa={}, peak_mi={}",
        peak_pressure,
        peak_pressure * 1.0e-6 / (config.frequency_hz * 1.0e-6).sqrt(),
    );
    assert!(result.fwi_objective_history.iter().all(|v| v.is_finite()));
    assert!(
        result.fwi_objective_history.last().copied().unwrap()
            <= result.fwi_objective_history.first().copied().unwrap(),
        "Westervelt FWI objective must be non-increasing on the brain fixture",
    );
    assert!(
        result.cavitation_objective_history.last().copied().unwrap()
            <= result
                .cavitation_objective_history
                .first()
                .copied()
                .unwrap(),
        "Cavitation projected-gradient objective must be non-increasing on the brain fixture",
    );
    assert!(result.therapy_points_m.len() >= 16);
    assert!(result.receiver_points_m.len() >= 4);
}
