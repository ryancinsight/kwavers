use super::*;
use crate::core::constants::cavitation::{SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER};
use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use crate::core::constants::numerical::MHZ_TO_HZ;
use ndarray::Array3;

fn detector() -> TherapyCavitationDetector {
    TherapyCavitationDetector::new(MHZ_TO_HZ, 0.0)
}

#[test]
fn test_blake_threshold_1um_value() {
    let det = detector();
    let expected =
        (ATMOSPHERIC_PRESSURE + VAPOR_PRESSURE_WATER - 2.0 * SURFACE_TENSION_WATER / 1e-6).abs();
    assert!(
        (det.blake_threshold - expected).abs() < 1.0,
        "Blake threshold {:.1} Pa ≠ expected {expected:.1} Pa",
        det.blake_threshold
    );
}

#[test]
fn test_blake_threshold_positive() {
    let det = detector();
    assert!(
        det.blake_threshold > 0.0,
        "Blake threshold must be positive; got {}",
        det.blake_threshold
    );
}

#[test]
fn test_blake_threshold_larger_nucleus_higher_pressure() {
    let det_1um = TherapyCavitationDetector::new_with_radius(MHZ_TO_HZ, 1e-6);
    let det_10um = TherapyCavitationDetector::new_with_radius(MHZ_TO_HZ, 10e-6);
    assert!(
        det_10um.blake_threshold > det_1um.blake_threshold,
        "10 µm nucleus should have higher Blake threshold than 1 µm"
    );
}

#[test]
fn test_minnaert_frequency_1um() {
    let det = detector();
    let f0 = det.minnaert_frequency(1e-6);
    assert!(
        (f0 - 3.26e6).abs() / 3.26e6 < 0.05,
        "Minnaert f₀(1µm) = {:.3e} Hz, expected ~3.26 MHz",
        f0
    );
}

#[test]
fn test_minnaert_frequency_scales_inversely_with_radius() {
    let det = detector();
    let f1 = det.minnaert_frequency(1e-6);
    let f2 = det.minnaert_frequency(2e-6);
    let ratio = f1 / f2;
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "f₀(R) should scale as 1/R: f(1µm)/f(2µm) = {ratio:.6}, expected 2.0"
    );
}

#[test]
fn test_threshold_detection_no_cavitation_below_threshold() {
    let det = detector();
    let p = Array3::from_elem((4, 4, 4), -0.5 * det.blake_threshold);
    let cav = det.detect(&p);
    assert!(
        cav.iter().all(|&c| !c),
        "pressure below Blake threshold should give no cavitation"
    );
}

#[test]
fn test_threshold_detection_cavitation_above_threshold() {
    let det = detector();
    let p = Array3::from_elem((4, 4, 4), -2.0 * det.blake_threshold);
    let cav = det.detect(&p);
    assert!(
        cav.iter().all(|&c| c),
        "pressure above Blake threshold should give cavitation everywhere"
    );
}

#[test]
fn test_threshold_detection_spatial_heterogeneity() {
    let det = detector();
    let p_high = -2.0 * det.blake_threshold;
    let p_low = -0.1 * det.blake_threshold;
    let mut p = Array3::from_elem((2, 2, 2), p_low);
    p[[0, 0, 0]] = p_high;
    p[[1, 1, 1]] = p_high;
    let cav = det.detect(&p);
    assert!(cav[[0, 0, 0]], "voxel 0,0,0 should cavitate");
    assert!(cav[[1, 1, 1]], "voxel 1,1,1 should cavitate");
    assert!(!cav[[0, 0, 1]], "voxel 0,0,1 should not cavitate");
}

#[test]
fn test_spectral_detection_zero_pressure_no_cavitation() {
    let mut det = TherapyCavitationDetector::new(3.26 * MHZ_TO_HZ, 0.0);
    det.method = CavitationDetectionMethod::Spectral;
    let p = Array3::zeros((4, 4, 4));
    let cav = det.detect(&p);
    assert!(
        cav.iter().all(|&c| !c),
        "zero pressure must never trigger cavitation"
    );
}

#[test]
fn test_spectral_detection_resonance_lowers_threshold() {
    let det_far = TherapyCavitationDetector::new(100e3, 0.0);
    let mut det_near = TherapyCavitationDetector::new(3.26 * MHZ_TO_HZ, 0.0);
    det_near.method = CavitationDetectionMethod::Spectral;

    let p_test = -0.80 * det_near.blake_threshold;
    let p = Array3::from_elem((4, 4, 4), p_test);

    let cav_far = det_far.detect(&p);
    let cav_near = det_near.detect(&p);

    assert!(
        cav_far.iter().all(|&c| !c),
        "off-resonance: 80% of threshold should not cavitate"
    );
    assert!(
        cav_near.iter().all(|&c| c),
        "at resonance: 80% of threshold should cavitate"
    );
}

#[test]
fn test_cavitation_index_at_threshold_is_one() {
    let det = detector();
    let ci = det.cavitation_index(det.blake_threshold);
    assert!(
        (ci - 1.0).abs() < 1e-12,
        "CI at P_Blake must equal 1.0; got {ci:.6e}"
    );
}

#[test]
fn test_cavitation_index_zero_pressure_is_zero() {
    let det = detector();
    assert_eq!(det.cavitation_index(0.0), 0.0);
}

#[test]
fn test_stable_cavitation_in_correct_range() {
    let det = detector();
    assert!(det.is_stable_cavitation(0.7 * det.blake_threshold));
    assert!(!det.is_stable_cavitation(0.4 * det.blake_threshold));
    assert!(!det.is_stable_cavitation(1.1 * det.blake_threshold));
}

#[test]
fn test_inertial_cavitation_above_threshold() {
    let det = detector();
    assert!(det.is_inertial_cavitation(det.blake_threshold));
    assert!(det.is_inertial_cavitation(2.0 * det.blake_threshold));
    assert!(!det.is_inertial_cavitation(0.9 * det.blake_threshold));
}

#[test]
fn test_cavitation_probability_at_threshold_is_half() {
    let det = detector();
    let p = det.cavitation_probability(det.blake_threshold);
    assert!(
        (p - 0.5).abs() < 1e-10,
        "probability at CI=1 must be 0.5; got {p:.6e}"
    );
}

#[test]
fn test_cavitation_probability_monotone() {
    let det = detector();
    let p1 = det.cavitation_probability(0.5 * det.blake_threshold);
    let p2 = det.cavitation_probability(1.0 * det.blake_threshold);
    let p3 = det.cavitation_probability(2.0 * det.blake_threshold);
    assert!(
        p1 < p2 && p2 < p3,
        "probability must increase with pressure"
    );
}
