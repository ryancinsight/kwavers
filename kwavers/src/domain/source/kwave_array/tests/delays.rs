//! Focus delay and element setter tests for [`KWaveArray`].

use super::super::KWaveArray;

#[test]
fn test_kwave_array_setters_preserve_elements() {
    let mut array = KWaveArray::new();
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    array.set_frequency(2.0e6);
    array.set_sound_speed(1600.0);
    assert_eq!(array.num_elements(), 1);
    assert!((array.frequency() - 2.0e6).abs() < 1.0e-12);
    let delays = array.get_focus_delays((0.0, 0.0, 1.0));
    assert_eq!(delays.len(), 1);
    assert!((delays[0] - 1.0 / 1600.0).abs() < 1.0e-12);
}

#[test]
fn test_focus_delays() {
    let mut array = KWaveArray::with_params(1e6, 1500.0);
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    array.add_disc_element((0.01, 0.0, 0.0), 0.005, None);
    let delays = array.get_focus_delays((0.005, 0.0, 0.02));
    assert_eq!(delays.len(), 2);
    assert!(delays[0] > 0.0);
    assert!(delays[1] > 0.0);
}

/// `get_element_delays` returns zero for both elements of a symmetric two-element array.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_get_element_delays_symmetric_array() {
    let mut array = KWaveArray::with_params(1e6, 1500.0);
    array.add_disc_element((-0.005, 0.0, 0.0), 0.002, None);
    array.add_disc_element((0.005, 0.0, 0.0), 0.002, None);
    let delays = array.get_element_delays((0.0, 0.0, 0.02));
    assert_eq!(delays.len(), 2);
    assert!(
        delays[0].abs() < 1e-12 && delays[1].abs() < 1e-12,
        "symmetric elements should have equal (zero) delays: {delays:?}"
    );
}

/// All delays are non-negative and the minimum delay is exactly 0.
/// # Panics
/// - Panics if assertion fails: `all delays must be non-negative`.
///
#[test]
fn test_get_element_delays_non_negative_min_zero() {
    let mut array = KWaveArray::with_params(1e6, 1500.0);
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    array.add_disc_element((0.01, 0.0, 0.0), 0.005, None);
    array.add_disc_element((0.02, 0.0, 0.0), 0.005, None);
    let delays = array.get_element_delays((0.01, 0.0, 0.03));
    assert_eq!(delays.len(), 3);
    for &d in &delays {
        assert!(d >= 0.0, "all delays must be non-negative");
    }
    let min_delay = delays.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(min_delay.abs() < 1e-15, "minimum delay must be 0");
}
