#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use crate::physics::acoustics::imaging::registration::engine::ImageRegistration;
    use crate::physics::acoustics::imaging::registration::spatial::SpatialTransform;

    #[test]
    fn test_rigid_registration_landmarks() {
        let registration = ImageRegistration::default();

        // Create simple landmark sets (translated by [1, 2, 3])
        let fixed_landmarks =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .unwrap();

        let moving_landmarks =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 3.0, 3.0])
                .unwrap();

        let result = registration
            .rigid_registration_landmarks(&fixed_landmarks, &moving_landmarks)
            .unwrap();

        // Check that we got a rigid body transform
        match result.spatial_transform {
            Some(SpatialTransform::RigidBody { translation, .. }) => {
                // Translation should be approximately [-1, -2, -3] to align centroids
                assert!((translation[0] + 1.0).abs() < 0.1);
                assert!((translation[1] + 2.0).abs() < 0.1);
                assert!((translation[2] + 3.0).abs() < 0.1);
            }
            _ => panic!("Expected RigidBody transform"),
        }

        // FRE should be small for perfect alignment
        assert!(result.quality_metrics.fre.unwrap() < 0.1);

        // Confidence should be high
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_temporal_synchronization() {
        let registration = ImageRegistration::default();

        // Create reference and target signals with known phase offset
        let n_samples = 1000;
        let sampling_rate = 1000.0; // 1 kHz
        let ref_signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin())
                .collect(),
        );
        let target_signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    (2.0 * std::f64::consts::PI * i as f64 / 100.0 + std::f64::consts::PI / 4.0)
                        .sin()
                })
                .collect(),
        );

        let sync = registration
            .temporal_synchronization(&ref_signal, &target_signal, sampling_rate)
            .unwrap();

        // Phase offset should be reasonable (cross-correlation result)
        assert!(sync.phase_offset.abs() < 2.0 * std::f64::consts::PI);

        // Quality metrics should be computed
        assert!(sync.quality_metrics.rms_timing_error >= 0.0);
        assert!(sync.quality_metrics.phase_lock_stability >= 0.0);
        assert!(sync.quality_metrics.phase_lock_stability <= 1.0);
        assert!(sync.quality_metrics.sync_success_rate >= 0.0);
        assert!(sync.quality_metrics.sync_success_rate <= 1.0);
    }

    #[test]
    fn test_registration_quality_metrics() {
        let registration = ImageRegistration::default();

        let fixed = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let moving = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let result = registration
            .rigid_registration_landmarks(&fixed, &moving)
            .unwrap();

        // For identical point sets, FRE should be very small
        assert!(result.quality_metrics.fre.unwrap() < 1e-10);

        // Confidence should be very high
        assert!(result.confidence > 0.99);
    }
}
