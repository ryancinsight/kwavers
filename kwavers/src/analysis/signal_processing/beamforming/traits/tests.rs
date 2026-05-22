//! Trait-conformance tests using a mock beamformer.

use ndarray::Array2;

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use super::{Beamformer, TimeDomainBeamformer};
use crate::core::error::KwaversResult;

// Mock beamformer for trait validation
struct MockBeamformer {
    sensor_count: usize,
    sampling_rate: f64,
    sound_speed: f64,
}

impl Beamformer for MockBeamformer {
    type Input = f64;
    type Output = f64;

    fn focus_at_point(&self, data: &Array2<f64>, focal_point: [f64; 3]) -> KwaversResult<f64> {
        self.validate_input(data)?;
        if !focal_point.iter().all(|x| x.is_finite()) {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Focal point contains non-finite values".into(),
            ));
        }
        Ok(data.sum())
    }

    fn expected_sensor_count(&self) -> usize {
        self.sensor_count
    }
}

impl TimeDomainBeamformer for MockBeamformer {
    fn sampling_rate(&self) -> f64 {
        self.sampling_rate
    }

    fn sound_speed(&self) -> f64 {
        self.sound_speed
    }

    fn compute_delay(
        &self,
        focal_point: [f64; 3],
        sensor_position: [f64; 3],
    ) -> KwaversResult<f64> {
        let dx = focal_point[0] - sensor_position[0];
        let dy = focal_point[1] - sensor_position[1];
        let dz = focal_point[2] - sensor_position[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        Ok(distance / self.sound_speed)
    }
}

#[test]
fn test_beamformer_trait() {
    let beamformer = MockBeamformer {
        sensor_count: 4,
        sampling_rate: 10e6,
        sound_speed: SOUND_SPEED_TISSUE,
    };

    assert_eq!(beamformer.expected_sensor_count(), 4);

    let data = Array2::<f64>::zeros((4, 100));
    let focal_point = [0.0, 0.0, 0.01];

    let focused = beamformer.focus_at_point(&data, focal_point).unwrap();
    // data is all-zeros so sum = 0.0
    assert_eq!(focused, 0.0);
}

#[test]
fn test_time_domain_beamformer_trait() {
    let beamformer = MockBeamformer {
        sensor_count: 8,
        sampling_rate: 10e6,
        sound_speed: SOUND_SPEED_TISSUE,
    };

    assert_eq!(beamformer.sampling_rate(), 10e6);
    assert_eq!(beamformer.sound_speed(), SOUND_SPEED_TISSUE);

    let focal = [0.0, 0.0, 0.02];
    let sensor = [0.001, 0.0, 0.0];
    let delay = beamformer.compute_delay(focal, sensor).unwrap();

    // Expected: distance ~= sqrt(0.001^2 + 0.02^2) = 0.020025 m
    // delay = 0.020025 / 1540 ≈ 1.3e-5 s
    assert!(delay > 0.0 && delay < 1e-4);
}

#[test]
fn test_apodization_default() {
    let beamformer = MockBeamformer {
        sensor_count: 8,
        sampling_rate: 10e6,
        sound_speed: SOUND_SPEED_TISSUE,
    };

    // Default apodization is uniform (1.0)
    for i in 0..8 {
        assert_eq!(beamformer.apodization_weight(i), 1.0);
    }
}

#[test]
fn test_trait_object_compatibility() {
    let beamformer: Box<dyn TimeDomainBeamformer> = Box::new(MockBeamformer {
        sensor_count: 4,
        sampling_rate: 10e6,
        sound_speed: SOUND_SPEED_TISSUE,
    });

    assert_eq!(beamformer.expected_sensor_count(), 4);
    assert_eq!(beamformer.sampling_rate(), 10e6);
}
