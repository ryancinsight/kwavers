use super::tracker::IntensityTracker;
use ndarray::Array3;

#[test]
fn test_tracker_creation() {
    let tracker = IntensityTracker::new(0.1, 1e-6).unwrap();
    assert_eq!(tracker.sample_count(), 0);
    assert_eq!(tracker.peak_intensity_w_cm2(), 0.0);
}

#[test]
fn test_invalid_parameters() {
    assert!(IntensityTracker::new(-0.1, 1e-6).is_err());
    assert!(IntensityTracker::new(0.1, -1e-6).is_err());
    assert!(IntensityTracker::new(1e-7, 1e-6).is_err()); // dt > window
}

#[test]
fn test_intensity_recording() {
    let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

    // Create simple test fields
    let pressure = Array3::from_elem([8, 8, 8], 1e6); // 1 MPa
    let impedance = Array3::from_elem([8, 8, 8], 1.5e6); // Water impedance

    let metrics = tracker
        .record_intensity(&pressure, &impedance, 0.0)
        .unwrap();

    // I = p²/Z = (1e6)² / 1.5e6 = 666,666 W/m² ≈ 666.7 kW/m²
    // spta is stored in W/m² (SI); 600_000 < 666,666 < 700_000
    assert!(metrics.spta > 6e5);
    assert!(metrics.spta < 7e5);
}

#[test]
fn test_peak_tracking() {
    let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

    let pressure = Array3::from_elem([4, 4, 4], 1e6);
    let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

    tracker
        .record_intensity(&pressure, &impedance, 0.0)
        .unwrap();
    let peak1 = tracker.peak_intensity_w_cm2();

    // Increase pressure
    let pressure2 = Array3::from_elem([4, 4, 4], 2e6);
    tracker
        .record_intensity(&pressure2, &impedance, 1e-6)
        .unwrap();
    let peak2 = tracker.peak_intensity_w_cm2();

    assert!(peak2 > peak1);
}

#[test]
fn test_thermal_dose() {
    let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();
    let pressure = Array3::from_elem([4, 4, 4], 0.0);
    let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

    // Record baseline
    tracker
        .record_intensity(&pressure, &impedance, 0.0)
        .unwrap();

    // Update thermal dose at elevated temperature
    let temperature = Array3::from_elem([4, 4, 4], 45.0); // 45°C
    tracker.update_thermal_dose(&temperature, 1.0).unwrap(); // 1 second

    let dose = tracker.thermal_dose();
    assert!(dose.cem43 > 0.0);
    assert_eq!(dose.current_temperature, 45.0);
    assert!(tracker.is_thermal_safe());
}

#[test]
fn test_spta_units() {
    let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

    let pressure = Array3::from_elem([4, 4, 4], 1e6); // 1 MPa
    let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

    tracker
        .record_intensity(&pressure, &impedance, 0.0)
        .unwrap();

    let spta_wm2 = tracker.metrics().spta;
    let spta_wcm2 = tracker.spta_w_cm2();

    // 1 W/cm² = 1e4 W/m²
    assert!((spta_wcm2 - spta_wm2 / 1e4).abs() < 0.01);
}

#[test]
fn test_reset() {
    let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

    let pressure = Array3::from_elem([4, 4, 4], 1e6);
    let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

    tracker
        .record_intensity(&pressure, &impedance, 0.0)
        .unwrap();
    assert!(tracker.sample_count() > 0);

    tracker.reset();
    assert_eq!(tracker.sample_count(), 0);
    assert_eq!(tracker.peak_intensity_w_cm2(), 0.0);
    assert_eq!(tracker.thermal_dose().cem43, 0.0);
}
