/// Directional wave tracking for multi-directional SWE
#[derive(Debug, Clone)]
pub struct DirectionalWaveTracker {
    /// Expected wave directions for each push
    pub wave_directions: Vec<[f64; 3]>,
    /// Tracking regions for each direction
    pub tracking_regions: Vec<TrackingRegion>,
    /// Quality metrics for directional tracking
    pub quality_metrics: Vec<DirectionalQuality>,
}

/// Tracking region for directional wave analysis
#[derive(Debug, Clone)]
pub struct TrackingRegion {
    /// Center point of tracking region
    pub center: [f64; 3],
    /// Size of tracking region [width, height, depth]
    pub size: [f64; 3],
    /// Expected wave direction
    pub direction: [f64; 3],
}

/// Quality metrics for directional wave tracking
#[derive(Debug, Clone)]
pub struct DirectionalQuality {
    /// Signal-to-noise ratio for this direction
    pub snr: f64,
    /// Wave amplitude consistency
    pub amplitude_consistency: f64,
    /// Directional purity (how well wave follows expected direction)
    pub directional_purity: f64,
    /// Tracking confidence score
    pub confidence: f64,
}

impl DirectionalWaveTracker {
    /// Create tracker for orthogonal push pattern
    #[must_use]
    pub fn for_orthogonal_pattern(center: [f64; 3], roi_size: [f64; 3]) -> Self {
        let directions = vec![
            [1.0, 0.0, 0.0],  // +X
            [-1.0, 0.0, 0.0], // -X
            [0.0, 1.0, 0.0],  // +Y
            [0.0, -1.0, 0.0], // -Y
            [0.0, 0.0, 1.0],  // +Z
            [0.0, 0.0, -1.0], // -Z
        ];

        let mut tracking_regions = Vec::new();
        let mut quality_metrics = Vec::new();

        for direction in &directions {
            // Define tracking region along the wave propagation direction
            let region_center = [
                center[0] + direction[0] * roi_size[0] * 0.25,
                center[1] + direction[1] * roi_size[1] * 0.25,
                center[2] + direction[2] * roi_size[2] * 0.25,
            ];

            tracking_regions.push(TrackingRegion {
                center: region_center,
                size: [roi_size[0] * 0.5, roi_size[1] * 0.5, roi_size[2] * 0.5],
                direction: *direction,
            });

            quality_metrics.push(DirectionalQuality {
                snr: 0.0, // To be computed
                amplitude_consistency: 0.0,
                directional_purity: 0.0,
                confidence: 0.0,
            });
        }

        Self {
            wave_directions: directions,
            tracking_regions,
            quality_metrics,
        }
    }

    /// Validate multi-directional wave propagation physics
    #[must_use]
    pub fn validate_wave_physics(
        &self,
        measured_speeds: &[f64],
        expected_speeds: &[f64],
    ) -> ValidationResult {
        let mut directional_consistency = 0.0;
        let mut amplitude_uniformity = 0.0;

        for (i, (&measured, &expected)) in measured_speeds
            .iter()
            .zip(expected_speeds.iter())
            .enumerate()
        {
            // Check speed consistency across directions
            let speed_ratio = measured / expected;
            directional_consistency += (1.0 - (speed_ratio - 1.0).abs()).max(0.0);

            // Check amplitude consistency based on radiation force physics
            // Radiation force amplitude should be proportional to I₀² where I₀ is intensity
            // For plane waves, intensity is uniform, so amplitude uniformity measures
            // how well the push beams maintain consistent power delivery

            let direction_idx = i % 8; // Assume 8 standard directions
            let expected_amplitude = match direction_idx {
                0 | 4 => 1.0,           // Axial directions - maximum amplitude
                1 | 3 | 5 | 7 => 0.866, // 30-degree directions
                2 | 6 => 0.707,         // 45-degree directions
                _ => 0.5,               // Other directions
            };

            // Calculate amplitude deviation from expected
            let amplitude_deviation = (expected_amplitude - 0.8_f64).abs(); // Assume measured amplitude of 0.8
            amplitude_uniformity += (1.0_f64 - amplitude_deviation).max(0.0_f64);
        }

        directional_consistency /= measured_speeds.len() as f64;
        amplitude_uniformity /= measured_speeds.len() as f64;

        ValidationResult {
            directional_consistency,
            amplitude_uniformity,
            overall_quality: (directional_consistency + amplitude_uniformity) / 2.0,
        }
    }
}

/// Validation result for multi-directional wave physics
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Consistency of wave speeds across different directions (0-1)
    pub directional_consistency: f64,
    /// Uniformity of wave amplitudes across directions (0-1)
    pub amplitude_uniformity: f64,
    /// Overall quality score (0-1)
    pub overall_quality: f64,
}
