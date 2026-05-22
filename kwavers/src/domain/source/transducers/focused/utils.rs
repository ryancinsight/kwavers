//! Utility functions for focused transducers
//!
//! Provides helper functions for creating common transducer configurations.

use super::bowl::{BowlConfig, BowlTransducer};
use super::multi_bowl::MultiBowlArray;
use crate::core::error::KwaversResult;

pub use super::multi_bowl::ApodizationType;

/// Helper function to create a spherical section bowl
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
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
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
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
    use crate::domain::grid::Grid;
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

    /// Test multi-bowl array with phase shifts (COMPREHENSIVE - Tier 3)
    ///
    /// This test generates sources on a 32³ grid with phase shifts.
    /// Execution time: >60s, classified as Tier 3 comprehensive validation.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    #[ignore = "Tier 3: Comprehensive validation (>60s execution time)"]
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
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();

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

    /// Test multi-bowl array with phase shifts (FAST - Tier 1)
    ///
    /// Fast version with reduced grid (8³) for CI/CD smoke test.
    /// Execution time: <1s, classified as Tier 1 fast validation.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_multi_bowl_phases_fast() {
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
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();

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
        use approx::assert_relative_eq;

        let bowl = make_bowl(0.064, 0.064, [0.0, 0.0, 0.0], 1e6, 1e6).unwrap();

        let focus_distance = 0.064; // radius of curvature = geometric focus [m]
        let frequency = 1e6;
        let c = crate::core::constants::SOUND_SPEED_WATER;
        let a = bowl.config.diameter / 2.0;       // aperture radius = 0.032 m
        let r = bowl.config.radius_of_curvature;  // 0.064 m
        let k = 2.0 * PI * frequency / c;
        let h = r - (r * r - a * a).sqrt(); // spherical-cap height

        // O'Neil (1949) on-axis amplitude at axial position z:
        //   |P(z)| = P₀ · 2 · |sin(k (d₂ − d₁) / 2)| / d₁
        // where d₁ = z, d₂ = √((z − (r−h))² + a²).
        //
        // Note: the argument is k(d₂−d₁)/2, NOT kh/2.  At z = focus = r
        // these differ because d₂ = √(h² + a²) ≠ h.  The absolute value of
        // sin is required: sin(k(d₂−d₁)/2) can be negative for many geometries.
        let f = r - h; // distance from bowl apex to focus along axis
        let d2_focus = ((focus_distance - f).powi(2) + a * a).sqrt();
        let phase_at_focus = k * (d2_focus - focus_distance);
        let expected_amplitude =
            bowl.config.amplitude * 2.0 * (phase_at_focus / 2.0).sin().abs() / focus_distance;

        // Peak pressure at z occurs at t = d₁/c + T/4 (propagation delay + sin max).
        // Using t = T/4 only is wrong because it ignores the k·d₁ phase in
        //   P(z, t) = P_amplitude · sin(ω t − k d₁ + φ).
        let peak_time = |z: f64| z / c + 1.0 / (4.0 * frequency);

        // Amplitude at the focal point.
        let p_focus_max = bowl.oneil_solution(focus_distance, peak_time(focus_distance)).abs();
        assert_relative_eq!(
            p_focus_max,
            expected_amplitude,
            epsilon = 0.01 * expected_amplitude
        );

        // Focusing: near-field amplitude must exceed far-field amplitude.
        //
        // For this geometry (r=0.064 m, a=0.032 m, F# = 1.0), the on-axis O'Neil
        // field exhibits near-field interference maxima before the geometric focus.
        // The pressure maximum is NOT necessarily at z = r (the centre of curvature)
        // — it can appear earlier for strongly focused bowls.  The physically
        // correct "focusing" check is that the near-field peak (here at z ≈ 0.05 m)
        // exceeds the far-field amplitude (z = 0.20 m >> r), where 1/z spherical
        // spreading dominates.
        let amp_near_field = bowl.oneil_solution(0.05, peak_time(0.05)).abs();
        let amp_far_field = bowl.oneil_solution(0.20, peak_time(0.20)).abs();
        assert!(
            amp_near_field > amp_far_field,
            "Near-field amplitude ({amp_near_field:.3e}) must exceed far-field ({amp_far_field:.3e})"
        );

        // Temporal periodicity: P(t + T/2) = −P(t) at the focal point.
        let t0 = peak_time(focus_distance);
        let p_t0 = bowl.oneil_solution(focus_distance, t0);
        let p_t_half = bowl.oneil_solution(focus_distance, t0 + 1.0 / (2.0 * frequency));
        assert_relative_eq!(
            p_t0,
            -p_t_half,
            epsilon = 0.01 * expected_amplitude
        );
    }
}
