use super::compound::PlaneWaveCompound;
use super::config::PlaneWaveConfig;
use ndarray::Array2;
use num_complex::Complex;

#[test]
fn test_plane_wave_config_default() {
    let config = PlaneWaveConfig::default();
    assert_eq!(config.num_angles, 11);
    assert!(config.angle_range > 0.0);
    assert!(config.frequency > 0.0);
}

#[test]
fn test_plane_wave_creation() {
    let config = PlaneWaveConfig::default();
    let result = PlaneWaveCompound::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_angle_generation() {
    let config = PlaneWaveConfig::default();
    let num_angles = config.num_angles;
    let compounding = PlaneWaveCompound::new(config).unwrap();
    let angles = compounding.get_angles();
    assert_eq!(angles.len(), num_angles);
    // First angle negative, last positive (symmetric sweep)
    if num_angles > 1 {
        assert!(
            angles[0] < 0.0,
            "First angle {} should be negative",
            angles[0]
        );
        assert!(
            angles[num_angles - 1] > 0.0,
            "Last angle {} should be positive",
            angles[num_angles - 1]
        );
    }
}

#[test]
fn test_apodization_windows() {
    for window in &["hann", "hamming", "blackman", "rect"] {
        let cfg = PlaneWaveConfig {
            apodization: window.to_string(),
            ..Default::default()
        };
        let comp = PlaneWaveCompound::new(cfg).unwrap();
        let apod = comp.compute_apodization();
        assert!(!apod.is_empty());
        // All weights in [0, 1]
        for &w in &apod {
            assert!(
                (0.0..=1.0).contains(&w),
                "Window {window}: weight {w} out of [0, 1]"
            );
        }
    }
}

#[test]
fn test_plane_wave_field_generation() {
    let config = PlaneWaveConfig::default();
    let compounding = PlaneWaveCompound::new(config).unwrap();
    let result = compounding.generate_plane_wave(0);
    assert!(result.is_ok());
    let field = result.unwrap();
    assert_eq!(field.nrows(), compounding.num_axial);
    assert_eq!(field.ncols(), compounding.num_lateral);
}

#[test]
fn test_beamforming() {
    let config = PlaneWaveConfig::default();
    let compounding = PlaneWaveCompound::new(config).unwrap();

    let received = Array2::from_elem(
        (compounding.num_axial, compounding.num_lateral),
        Complex::new(1.0, 0.0),
    );

    let result = compounding.beamform_angle(0, &received);
    assert!(result.is_ok());
}

#[test]
fn test_frame_rate_estimate() {
    let config = PlaneWaveConfig::default();
    let num_angles = config.num_angles;
    let compounding = PlaneWaveCompound::new(config).unwrap();
    let (speedup, fps) = compounding.frame_rate_estimate();

    assert_eq!(speedup, num_angles as f64);
    assert!(fps > 30.0, "fps {fps} should exceed focused-beam baseline");
}

#[test]
fn test_process_frame() {
    let config = PlaneWaveConfig::default();
    let num_angles = config.num_angles;
    let mut compounding = PlaneWaveCompound::new(config).unwrap();

    let received_fields: Vec<_> = (0..num_angles)
        .map(|_| {
            Array2::from_elem(
                (compounding.num_axial, compounding.num_lateral),
                Complex::new(1.0, 0.0),
            )
        })
        .collect();

    let result = compounding.process_frame(&received_fields);
    assert!(result.is_ok());

    let image = result.unwrap();
    assert_eq!(image.nrows(), compounding.num_axial);
    assert_eq!(image.ncols(), compounding.num_lateral);

    // Display image must be normalized to [0, 1]
    for &v in image.iter() {
        assert!((0.0..=1.0).contains(&v), "Display pixel {v} out of [0, 1]");
    }
}
