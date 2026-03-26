#[cfg(test)]
mod tests {
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use crate::physics::acoustics::imaging::modalities::elastography::radiation_force::impulse::{AcousticRadiationForce, PushPulseParameters};
    use crate::physics::acoustics::imaging::modalities::elastography::radiation_force::patterns::MultiDirectionalPush;
    use crate::physics::acoustics::imaging::modalities::elastography::radiation_force::tracking::DirectionalWaveTracker;
    use crate::physics::acoustics::mechanics::elastic_wave::ElasticBodyForceConfig;
    use std::f64::consts::PI;

    #[test]
    fn test_push_parameters_default() {
        let params = PushPulseParameters::default();
        assert_eq!(params.frequency, 5.0e6);
        assert_eq!(params.duration, 150e-6);
        assert!((params.f_number - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_push_parameters_validation() {
        let result = PushPulseParameters::new(-1.0, 100e-6, 1000.0, 0.04, 2.0);
        assert!(result.is_err());

        let result = PushPulseParameters::new(5e6, -100e-6, 1000.0, 0.04, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_radiation_force_creation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let result = AcousticRadiationForce::new(&grid, &medium);
        assert!(result.is_ok());
    }

    #[test]
    fn test_push_pulse_generation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let arf = AcousticRadiationForce::new(&grid, &medium).unwrap();

        let push_location = [0.025, 0.025, 0.025];
        let displacement = arf.push_pulse_pseudo_displacement(push_location).unwrap();

        // Check displacement field properties
        assert_eq!(displacement.dim(), (50, 50, 50));

        // Maximum displacement should be at or near focal point
        let max_disp = displacement
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_disp > 0.0, "Maximum displacement should be positive");

        // Displacement should decay away from focal point
        let corner_disp = displacement[[0, 0, 0]];
        assert!(
            corner_disp < max_disp * 0.1,
            "Displacement should be localized"
        );
    }

    #[test]
    fn test_multi_directional_push_creation() {
        let center = [0.025, 0.025, 0.025];
        let spacing = 0.01;

        let pattern = MultiDirectionalPush::orthogonal_pattern(center, spacing);

        // Should have 6 pushes (3 axes × 2 directions)
        assert_eq!(pattern.pushes.len(), 6);
        assert_eq!(pattern.time_delays.len(), 6);

        // Check that pushes are positioned correctly
        let push_x_pos = pattern.pushes[0].location; // +X direction
        assert!((push_x_pos[0] - (center[0] + spacing)).abs() < 1e-10);
        assert!((push_x_pos[1] - center[1]).abs() < 1e-10);
        assert!((push_x_pos[2] - center[2]).abs() < 1e-10);
    }

    #[test]
    fn test_compound_push_pattern() {
        let center = [0.025, 0.025, 0.025];
        let radius = 0.015;
        let n_pushes = 8;

        let pattern = MultiDirectionalPush::compound_pattern(center, radius, n_pushes);

        assert_eq!(pattern.pushes.len(), n_pushes);

        // Check that pushes are distributed around the circle
        for (i, push) in pattern.pushes.iter().enumerate() {
            let expected_angle = 2.0 * PI * (i as f64) / (n_pushes as f64);
            let mut actual_angle =
                (push.location[1] - center[1]).atan2(push.location[0] - center[0]);
            // Normalize angle to [0, 2π) range
            if actual_angle < 0.0 {
                actual_angle += 2.0 * PI;
            }
            // Allow for small numerical differences and account for atan2 range
            let angle_diff = (actual_angle - expected_angle).abs().min(
                (actual_angle - expected_angle + 2.0 * PI)
                    .abs()
                    .min((actual_angle - expected_angle - 2.0 * PI).abs()),
            );
            assert!(
                angle_diff < 0.1,
                "Push {}: expected angle {:.3}, got {:.3}",
                i,
                expected_angle,
                actual_angle
            );
        }
    }

    #[test]
    fn test_focused_push_pattern() {
        let roi_center = [0.025, 0.025, 0.025];
        let roi_size = [0.02, 0.02, 0.02];
        let density = 10;

        let pattern = MultiDirectionalPush::focused_pattern(roi_center, roi_size, density);

        // Should not exceed density limit
        assert!(pattern.pushes.len() <= density);

        // All pushes should be within ROI bounds
        for push in &pattern.pushes {
            assert!(push.location[0] >= roi_center[0] - roi_size[0] / 2.0);
            assert!(push.location[0] <= roi_center[0] + roi_size[0] / 2.0);
            assert!(push.location[1] >= roi_center[1] - roi_size[1] / 2.0);
            assert!(push.location[1] <= roi_center[1] + roi_size[1] / 2.0);
            assert!(push.location[2] >= roi_center[2] - roi_size[2] / 2.0);
            assert!(push.location[2] <= roi_center[2] + roi_size[2] / 2.0);
        }
    }

    #[test]
    fn test_directional_wave_tracker() {
        let center = [0.025, 0.025, 0.025];
        let roi_size = [0.04, 0.04, 0.04];

        let tracker = DirectionalWaveTracker::for_orthogonal_pattern(center, roi_size);

        // Should have 6 directions and tracking regions
        assert_eq!(tracker.wave_directions.len(), 6);
        assert_eq!(tracker.tracking_regions.len(), 6);
        assert_eq!(tracker.quality_metrics.len(), 6);

        // Check that directions are orthogonal unit vectors
        for direction in &tracker.wave_directions {
            let magnitude =
                (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
            assert!(
                (magnitude - 1.0).abs() < 1e-10,
                "Direction should be unit vector"
            );
        }
    }

    #[test]
    fn test_wave_physics_validation() {
        let center = [0.025, 0.025, 0.025];
        let roi_size = [0.04, 0.04, 0.04];

        let tracker = DirectionalWaveTracker::for_orthogonal_pattern(center, roi_size);

        // Simulate measured and expected speeds
        let measured_speeds = vec![3.0, 2.9, 3.1, 3.0, 2.8, 3.2];
        let expected_speeds = vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0];

        let result = tracker.validate_wave_physics(&measured_speeds, &expected_speeds);

        // Should have reasonable validation scores
        assert!(result.directional_consistency >= 0.0 && result.directional_consistency <= 1.0);
        assert!(result.amplitude_uniformity >= 0.0 && result.amplitude_uniformity <= 1.0);
        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
    }

    #[test]
    fn test_multi_directional_push_application() {
        let grid = Grid::new(30, 30, 30, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let arf = AcousticRadiationForce::new(&grid, &medium).unwrap();

        let center = [0.015, 0.015, 0.015];
        let pattern = MultiDirectionalPush::orthogonal_pattern(center, 0.005);

        let forces = arf
            .multi_directional_body_forces(&pattern)
            .expect("multi-directional ARFI must yield body-force configs");

        assert_eq!(forces.len(), pattern.pushes.len());
        for cfg in &forces {
            match cfg {
                ElasticBodyForceConfig::GaussianImpulse {
                    impulse_n_per_m3_s, ..
                } => {
                    assert!(*impulse_n_per_m3_s > 0.0);
                }
            }
        }
    }
}
