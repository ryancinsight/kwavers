use super::compound::PlaneWaveCompound;
use super::config::PlaneWaveCompoundingConfig;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use ndarray::Array2;
use num_complex::Complex;

#[test]
fn test_plane_wave_config_default() {
    let config = PlaneWaveCompoundingConfig::default();
    assert_eq!(config.num_angles, 11);
    assert!(config.angle_range > 0.0);
    assert!(config.frequency > 0.0);
}

#[test]
fn test_plane_wave_creation() {
    let config = PlaneWaveCompoundingConfig::default();
    let result = PlaneWaveCompound::new(config);
    let compounding = result.expect("valid plane-wave config");
    assert_eq!(compounding.num_angles(), 11);
    assert_eq!(compounding.dimensions(), (200, 80));
}

#[test]
fn test_angle_generation() {
    let config = PlaneWaveCompoundingConfig::default();
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
        let cfg = PlaneWaveCompoundingConfig {
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
    let config = PlaneWaveCompoundingConfig::default();
    let compounding = PlaneWaveCompound::new(config).unwrap();
    let field = compounding.generate_plane_wave(0).unwrap();
    assert_eq!(field.nrows(), compounding.num_axial);
    assert_eq!(field.ncols(), compounding.num_lateral);
}

#[test]
fn test_beamforming() {
    let config = PlaneWaveCompoundingConfig::default();
    let compounding = PlaneWaveCompound::new(config).unwrap();

    let received = Array2::from_elem(
        (compounding.num_axial, compounding.num_lateral),
        Complex::new(1.0, 0.0),
    );

    let beamformed = compounding.beamform_angle(0, &received).unwrap();
    assert_eq!(beamformed.dim(), received.dim());
}

#[test]
fn test_frame_rate_estimate() {
    let config = PlaneWaveCompoundingConfig::default();
    let num_angles = config.num_angles;
    let compounding = PlaneWaveCompound::new(config).unwrap();
    let (speedup, fps) = compounding.frame_rate_estimate();

    assert_eq!(speedup, num_angles as f64);
    assert!(fps > 30.0, "fps {fps} should exceed focused-beam baseline");
}

#[test]
fn test_process_frame() {
    let config = PlaneWaveCompoundingConfig::default();
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

    let image = compounding.process_frame(&received_fields).unwrap();
    assert_eq!(image.nrows(), compounding.num_axial);
    assert_eq!(image.ncols(), compounding.num_lateral);

    // Display image must be normalized to [0, 1]
    for &v in image.iter() {
        assert!((0.0..=1.0).contains(&v), "Display pixel {v} out of [0, 1]");
    }
}

#[test]
fn test_thermal_acoustic_config_uses_plane_wave_geometry() {
    let plane_wave = PlaneWaveCompoundingConfig {
        sound_speed: SOUND_SPEED_WATER_SIM,
        aperture_size: 0.012,
        lateral_step: 0.001,
        element_spacing: 0.0005,
        depth: 0.020,
        axial_step: 0.002,
        ..Default::default()
    };
    let compounding = PlaneWaveCompound::new(plane_wave.clone()).unwrap();

    let thermal = compounding.config();

    assert_eq!(thermal.nx, 12);
    assert_eq!(thermal.ny, 1);
    assert_eq!(thermal.nz, 10);
    assert_eq!(thermal.dx, plane_wave.lateral_step);
    assert_eq!(thermal.dy, plane_wave.element_spacing);
    assert_eq!(thermal.dz, plane_wave.axial_step);
    assert_eq!(thermal.c_ref, plane_wave.sound_speed);
    assert_eq!(
        thermal.dt,
        0.3 * plane_wave.element_spacing / plane_wave.sound_speed
    );
}
