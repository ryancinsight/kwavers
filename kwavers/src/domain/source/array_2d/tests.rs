use super::array::TransducerArray2D;
use super::types::{ApodizationType, TransducerArray2DConfig};

fn create_test_config() -> TransducerArray2DConfig {
    TransducerArray2DConfig {
        number_elements: 16,
        element_width: 0.3e-3,
        element_length: 10e-3,
        element_spacing: 0.5e-3,
        radius: f64::INFINITY,
        center_position: (0.0, 0.0, 0.0),
    }
}

#[test]
fn test_array_creation() {
    let config = create_test_config();
    let array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

    assert_eq!(array.num_elements(), 16);
    assert!(array.satisfies_nyquist());
}

#[test]
fn test_focus_and_steering() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

    array.set_focus_distance(20e-3);
    array.set_steering_angle(10.0);

    assert!((array.focus_distance() - 20e-3).abs() < 1e-10);
    assert!((array.steering_angle() - 10.0).abs() < 1e-10);

    let positions = array.element_positions();
    assert_eq!(positions.len(), 16);
}

#[test]
fn test_apodization() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

    array.set_transmit_apodization(ApodizationType::Hanning);
    array.set_receive_apodization(ApodizationType::Hamming);
}

#[test]
fn test_active_elements() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

    let mut mask = vec![true; 16];
    for i in (0..16).step_by(2) {
        mask[i] = false;
    }

    array.set_active_elements(&mask).unwrap();

    let active = array.get_active_elements();
    assert_eq!(active.len(), 16);
    for i in (0..16).step_by(2) {
        assert!(!active[i]);
    }
}

#[test]
fn test_invalid_config() {
    let config = TransducerArray2DConfig {
        number_elements: 0,
        ..create_test_config()
    };

    assert!(TransducerArray2D::new(config, 1540.0, 1e6).is_err());
}

#[test]
fn test_aperture_calculation() {
    let config = create_test_config();
    let array = TransducerArray2D::new(config, 1540.0, 1e6).unwrap();

    let expected = 15.0 * 0.5e-3 + 0.3e-3;
    assert!((array.aperture_width() - expected).abs() < 1e-10);
}
