//! Tests for advanced ultrasound imaging techniques

use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use leto::Array3;

#[test]
fn test_synthetic_aperture_config() {
    let config = SyntheticApertureConfig::default();
    assert_eq!(config.num_tx_elements, 64);
    assert_eq!(config.num_rx_elements, 64);
    assert!((config.sound_speed - SOUND_SPEED_TISSUE).abs() < 1e-6);
}

#[test]
fn test_plane_wave_config() {
    let config = UltrasoundPlaneWaveConfig::default();
    assert_eq!(config.tx_angle, 0.0);
    assert_eq!(config.num_elements, 64);
    assert!((config.frequency - 5.0 * MHZ_TO_HZ).abs() < 1e-6);
}

#[test]
fn test_chirp_generation() {
    let config = CodedExcitationConfig {
        code: ExcitationCode::Chirp {
            start_freq: 2.0 * MHZ_TO_HZ,
            end_freq: 8.0 * MHZ_TO_HZ,
            length: 100,
        },
        sound_speed: SOUND_SPEED_TISSUE,
        sampling_frequency: 20.0 * MHZ_TO_HZ,
    };

    let processor = CodedExcitationProcessor::new(config);
    let code = processor.generate_code();

    assert_eq!(code.len(), 100);
    assert!(code.iter().any(|&x| x.norm() > 0.0));
}

#[test]
fn test_barker_generation() {
    let config = CodedExcitationConfig {
        code: ExcitationCode::Barker { length: 7 },
        sound_speed: SOUND_SPEED_TISSUE,
        sampling_frequency: 20.0 * MHZ_TO_HZ,
    };

    let processor = CodedExcitationProcessor::new(config);
    let code = processor.generate_code();

    assert_eq!(code.len(), 7);
    for &val in code.iter() {
        assert!((val.re - 1.0).abs() < 1e-6 || (val.re + 1.0).abs() < 1e-6);
        assert!(val.im.abs() < 1e-6);
    }
}

#[test]
fn test_snr_improvement_calculation() {
    let config = CodedExcitationConfig {
        code: ExcitationCode::Chirp {
            start_freq: 2.0 * MHZ_TO_HZ,
            end_freq: 8.0 * MHZ_TO_HZ,
            length: 100,
        },
        sound_speed: SOUND_SPEED_TISSUE,
        sampling_frequency: 20.0 * MHZ_TO_HZ,
    };

    let processor = CodedExcitationProcessor::new(config);
    let snr_improvement = processor.theoretical_snr_improvement();

    assert!((snr_improvement - 10.0).abs() < 1e-6);
}

#[test]
fn test_plane_wave_compounding() {
    let base_config = UltrasoundPlaneWaveConfig::default();
    let angles = vec![-10.0f64.to_radians(), 0.0, 10.0f64.to_radians()];
    let compounding = PlaneWaveCompounding::new(&angles, base_config);

    let height = 100;
    let width = 100;
    let mut images = Array3::<f64>::zeros((angles.len(), height, width));

    for angle in 0..angles.len() {
        for i in 0..height {
            for j in 0..width {
                images[[angle, i, j]] = (angle + 1) as f64;
            }
        }
    }

    let compounded = compounding.compound(&images);

    assert_eq!(compounded.shape()[0], height);
    assert_eq!(compounded.shape()[1], width);

    assert!((compounded[[50, 50]] - 2.0).abs() < 1e-6);
}
