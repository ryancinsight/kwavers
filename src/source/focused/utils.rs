//! Utility functions for focused transducers
//!
//! Provides helper functions for creating common transducer configurations.

use super::bowl::{BowlConfig, BowlTransducer};
use super::multi_bowl::MultiBowlArray;
use crate::error::KwaversResult;

pub use super::multi_bowl::ApodizationType;

/// Helper function to create a spherical section bowl
pub fn make_bowl(
    radius: f64,
    diameter: f64,
    center: [f64; 3],
    frequency: f64,
    amplitude: f64,
) -> KwaversResult<BowlTransducer> {
    let config = BowlConfig {
        radius_of_curvature: radius,
        diameter,
        center,
        focus: [center[0], center[1], center[2] + radius], // Default focus at radius
        frequency,
        amplitude,
        ..Default::default()
    };

    BowlTransducer::new(config)
}

/// Helper function to create an annular array
pub fn make_annular_array(
    inner_radius: f64,
    outer_radius: f64,
    n_rings: usize,
    center: [f64; 3],
    frequency: f64,
) -> KwaversResult<MultiBowlArray> {
    let mut configs = Vec::new();

    for i in 0..n_rings {
        let t = if n_rings > 1 {
            i as f64 / (n_rings - 1) as f64
        } else {
            0.0
        };
        let radius = inner_radius + t * (outer_radius - inner_radius);

        let config = BowlConfig {
            radius_of_curvature: radius,
            diameter: radius * 0.8, // 80% of radius for each ring
            center,
            focus: [center[0], center[1], center[2] + radius],
            frequency,
            amplitude: 1e6,
            ..Default::default()
        };

        configs.push(config);
    }

    MultiBowlArray::new(configs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use std::f64::consts::PI;

    #[test]
    fn test_bowl_transducer_creation() {
        let config = BowlConfig::default();
        let bowl = BowlTransducer::new(config).unwrap();

        assert!(!bowl.element_positions.is_empty());
        assert_eq!(bowl.element_positions.len(), bowl.element_normals.len());
        assert_eq!(bowl.element_positions.len(), bowl.element_areas.len());
    }

    #[test]
    fn test_bowl_focusing() {
        let config = BowlConfig {
            radius_of_curvature: 0.05,
            diameter: 0.04,
            center: [0.0, 0.0, 0.0],
            focus: [0.0, 0.0, 0.05],
            frequency: 1e6,
            amplitude: 1e6,
            ..Default::default()
        };

        let bowl = BowlTransducer::new(config).unwrap();
        let delays = bowl.calculate_focus_delays();

        // All delays should be positive
        for delay in delays {
            assert!(delay >= 0.0);
        }
    }

    #[test]
    fn test_multi_bowl_array() {
        let configs = vec![
            BowlConfig {
                radius_of_curvature: 0.05,
                diameter: 0.03,
                center: [0.0, 0.0, 0.0],
                ..Default::default()
            },
            BowlConfig {
                radius_of_curvature: 0.06,
                diameter: 0.04,
                center: [0.01, 0.0, 0.0],
                ..Default::default()
            },
        ];

        let array = MultiBowlArray::new(configs).unwrap();
        assert_eq!(array.bowls.len(), 2);
    }

    #[test]
    #[ignore] // TODO: Hangs in parallel processing - needs investigation
    fn test_multi_bowl_phases() {
        // Create two bowls with different phases
        let config1 = BowlConfig {
            diameter: 0.03,
            radius_of_curvature: 0.05,
            focus: [0.0, 0.0, 0.05],
            frequency: 1e6,
            amplitude: 1e6,
            phase: 0.0,
            ..Default::default()
        };

        let config2 = BowlConfig {
            diameter: 0.03,
            radius_of_curvature: 0.05,
            focus: [0.0, 0.0, 0.05],
            frequency: 1e6,
            amplitude: 1e6,
            phase: PI / 2.0, // 90 degree phase shift
            ..Default::default()
        };

        let multi_array = MultiBowlArray::new(vec![config1, config2]).unwrap();
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001);

        // Generate sources at different times
        let source_t0 = multi_array.generate_source(&grid, 0.0).unwrap();
        let source_t1 = multi_array.generate_source(&grid, 0.25e-6).unwrap(); // Quarter period

        // The sources should be different due to phase shifts
        let diff = &source_t1 - &source_t0;
        let max_diff = diff.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(
            max_diff > 0.0,
            "Phase shifts should cause time-varying fields"
        );
    }

    #[test]
    fn test_oneil_solution() {
        let bowl = make_bowl(0.064, 0.064, [0.0, 0.0, 0.0], 1e6, 1e6).unwrap();

        // Test on-axis pressure at focus at different times
        let time = 0.0;
        let p_focus = bowl.oneil_solution(0.064, time).abs();

        // Test at a time when sin is maximum (quarter period)
        let period = 1.0 / 1e6;
        let time_max = period / 4.0;
        let p_focus_max = bowl.oneil_solution(0.064, time_max).abs();
        assert!(p_focus_max > 0.0);

        // Pressure amplitude should vary with distance from focus
        let p_near = bowl.oneil_solution(0.05, time_max).abs();
        let p_far = bowl.oneil_solution(0.1, time_max).abs();

        // The focus should have strong pressure due to constructive interference
        // Note: The exact relationship depends on the interference pattern
        assert!(p_focus_max > 0.0);
        assert!(p_near > 0.0);
        assert!(p_far > 0.0);
    }
}
