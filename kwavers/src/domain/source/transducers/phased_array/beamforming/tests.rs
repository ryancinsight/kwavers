use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use crate::core::constants::numerical::MHZ_TO_HZ;

#[test]
fn test_focus_delays_symmetry() {
    let calculator = BeamformingCalculator::with_medium(SOUND_SPEED_TISSUE, MHZ_TO_HZ);

    // Symmetric linear array
    let positions = vec![
        (-1.5e-3, 0.0, 0.0),
        (-0.5e-3, 0.0, 0.0),
        (0.5e-3, 0.0, 0.0),
        (1.5e-3, 0.0, 0.0),
    ];

    let target = (0.0, 0.0, 0.05); // Axial focus

    let delays = calculator.calculate_focus_delays(&positions, target);

    assert_eq!(delays.len(), 4);

    // Symmetric array should have symmetric delays
    assert!((delays[0] - delays[3]).abs() < 1e-10);
    assert!((delays[1] - delays[2]).abs() < 1e-10);

    // All delays should be non-negative (normalized)
    for &delay in &delays {
        assert!(delay >= 0.0);
    }
}

#[test]
fn test_steering_delays_broadside() {
    let calculator = BeamformingCalculator::with_medium(SOUND_SPEED_TISSUE, MHZ_TO_HZ);

    let positions = vec![(0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0)];

    // Broadside: theta = 0 (axial)
    let delays = calculator.calculate_steering_delays(&positions, 0.0, 0.0);

    assert_eq!(delays.len(), 3);

    // All delays should be zero for broadside
    for &delay in &delays {
        assert!(delay.abs() < 1e-10);
    }
}

#[test]
fn test_plane_wave_delays() {
    let calculator = BeamformingCalculator::with_medium(SOUND_SPEED_TISSUE, MHZ_TO_HZ);

    let positions = vec![(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)];

    // Axial plane wave
    let direction = (0.0, 0.0, 1.0);
    let delays = calculator.calculate_plane_wave_delays(&positions, direction);

    assert_eq!(delays.len(), 2);

    // Broadside should have zero delays
    for &delay in &delays {
        assert!(delay.abs() < 1e-10);
    }
}

#[test]
fn test_beam_width_positive() {
    let calculator = BeamformingCalculator::with_medium(SOUND_SPEED_TISSUE, MHZ_TO_HZ);

    let beam_width = calculator.calculate_beam_width(0.01); // 1cm aperture

    assert!(beam_width > 0.0);
    assert!(beam_width.is_finite());
}

#[test]
fn test_focal_zone_positive() {
    let calculator = BeamformingCalculator::with_medium(SOUND_SPEED_TISSUE, MHZ_TO_HZ);

    let focal_zone = calculator.calculate_focal_zone(0.01, 0.05); // 1cm aperture, 5cm focus

    assert!(focal_zone > 0.0);
    assert!(focal_zone.is_finite());
}
