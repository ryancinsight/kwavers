//! Wavefront Analysis for Source Localization
//!
//! Detects and characterizes acoustic wavefronts to estimate source distance
//! and distinguish between plane waves (far-field) and spherical waves (near-field).

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Wavefront type detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WavefrontType {
    /// Spherical wavefront (near-field point source)
    Spherical,

    /// Plane wavefront (far-field or beam)
    Plane,

    /// Uncertain
    Unknown,
}

/// Wavefront analysis result
#[derive(Debug, Clone)]
pub struct WavefrontAnalysis {
    /// Detected wavefront type
    pub wavefront_type: WavefrontType,

    /// Estimated source distance [m] (for spherical waves)
    pub source_distance: Option<f64>,

    /// Wavefront propagation direction [x, y, z]
    pub propagation_direction: [f64; 3],

    /// Wavefront curvature [1/m]
    pub curvature: f64,

    /// Confidence in detection (0.0-1.0)
    pub confidence: f64,
}

/// Wavefront analyzer
#[derive(Debug)]
pub struct WavefrontAnalyzer {
    /// Grid spacing [m]
    grid_spacing: f64,

    /// Plane wave detection threshold
    plane_wave_threshold: f64,
}

impl WavefrontAnalyzer {
    /// Create new wavefront analyzer
    pub fn new(grid_spacing: f64) -> KwaversResult<Self> {
        if !grid_spacing.is_finite() || grid_spacing <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid grid spacing".to_string(),
            ));
        }

        Ok(Self {
            grid_spacing,
            plane_wave_threshold: 0.9,
        })
    }

    /// Detect wavefront type from pressure field
    pub fn detect_wavefront(
        &self,
        pressure_field: &Array3<f64>,
    ) -> KwaversResult<WavefrontAnalysis> {
        let (nx, ny, nz) = pressure_field.dim();

        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "Field must be at least 3x3x3".to_string(),
            ));
        }

        // Estimate spatial gradients
        let mut grad_mag_sq = 0.0;
        let mut grad_sum = [0.0_f64; 3];
        let mut laplacian = 0.0;
        let mut count = 0;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let center = pressure_field[[i, j, k]];

                    // Gradient magnitude squared
                    let dx = (pressure_field[[i + 1, j, k]] - pressure_field[[i - 1, j, k]])
                        / (2.0 * self.grid_spacing);
                    let dy = (pressure_field[[i, j + 1, k]] - pressure_field[[i, j - 1, k]])
                        / (2.0 * self.grid_spacing);
                    let dz = (pressure_field[[i, j, k + 1]] - pressure_field[[i, j, k - 1]])
                        / (2.0 * self.grid_spacing);

                    grad_mag_sq += dx * dx + dy * dy + dz * dz;
                    grad_sum[0] += dx;
                    grad_sum[1] += dy;
                    grad_sum[2] += dz;

                    // Laplacian (curvature)
                    let d2x = (pressure_field[[i + 1, j, k]] - 2.0 * center
                        + pressure_field[[i - 1, j, k]])
                        / (self.grid_spacing * self.grid_spacing);
                    let d2y = (pressure_field[[i, j + 1, k]] - 2.0 * center
                        + pressure_field[[i, j - 1, k]])
                        / (self.grid_spacing * self.grid_spacing);
                    let d2z = (pressure_field[[i, j, k + 1]] - 2.0 * center
                        + pressure_field[[i, j, k - 1]])
                        / (self.grid_spacing * self.grid_spacing);

                    laplacian += d2x + d2y + d2z;

                    count += 1;
                }
            }
        }

        if count == 0 {
            return Err(KwaversError::InvalidInput(
                "Cannot compute gradients in small field".to_string(),
            ));
        }

        grad_mag_sq /= count as f64;
        laplacian /= count as f64;

        // Estimate curvature: κ = |∇²p| / |∇p|^(3/2)
        let grad_mag = grad_mag_sq.sqrt();
        let curvature = if grad_mag > 1e-6 {
            laplacian.abs() / grad_mag.powf(1.5)
        } else {
            0.0
        };

        // Detect wavefront type
        let (wavefront_type, source_distance, confidence) = if curvature < self.plane_wave_threshold
        {
            // Plane wave: low curvature
            (WavefrontType::Plane, None, 0.8)
        } else {
            // Spherical wave: high curvature
            // Estimate distance from radius of curvature: R = 1/κ
            let distance = if curvature > 1e-6 {
                Some(1.0 / curvature)
            } else {
                None
            };
            (WavefrontType::Spherical, distance, 0.7)
        };

        // Estimate propagation direction from mean pressure gradient.
        // The wavefront propagates in the direction of ∇p (high→low pressure).
        let grad_norm =
            (grad_sum[0] * grad_sum[0] + grad_sum[1] * grad_sum[1] + grad_sum[2] * grad_sum[2])
                .sqrt();
        let propagation_direction = if grad_norm > 1e-30 {
            [
                grad_sum[0] / grad_norm,
                grad_sum[1] / grad_norm,
                grad_sum[2] / grad_norm,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        Ok(WavefrontAnalysis {
            wavefront_type,
            source_distance,
            propagation_direction,
            curvature,
            confidence,
        })
    }

    /// Detect plane waves in pressure field
    pub fn detect_plane_wave(&self, pressure_field: &Array3<f64>) -> KwaversResult<bool> {
        let analysis = self.detect_wavefront(pressure_field)?;
        Ok(analysis.wavefront_type == WavefrontType::Plane && analysis.confidence > 0.7)
    }

    /// Estimate source distance from spherical wavefront
    pub fn estimate_source_distance(
        &self,
        pressure_field: &Array3<f64>,
    ) -> KwaversResult<Option<f64>> {
        let analysis = self.detect_wavefront(pressure_field)?;
        Ok(analysis.source_distance)
    }

    /// Set plane wave detection threshold
    pub fn set_plane_wave_threshold(&mut self, threshold: f64) {
        self.plane_wave_threshold = threshold.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavefront_analyzer_creation() {
        let result = WavefrontAnalyzer::new(0.001);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wavefront_analyzer_invalid_spacing() {
        let result = WavefrontAnalyzer::new(-0.001);
        assert!(result.is_err());
    }

    #[test]
    fn test_wavefront_detection_small_field() {
        let analyzer = WavefrontAnalyzer::new(0.001).unwrap();
        let field = Array3::zeros((2, 2, 2));

        let result = analyzer.detect_wavefront(&field);
        assert!(result.is_err());
    }

    #[test]
    fn test_wavefront_detection_uniform_field() {
        let analyzer = WavefrontAnalyzer::new(0.001).unwrap();
        let field = Array3::ones((10, 10, 10));

        let result = analyzer.detect_wavefront(&field);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.wavefront_type, WavefrontType::Plane);
    }

    #[test]
    fn test_plane_wave_detection() {
        let analyzer = WavefrontAnalyzer::new(0.001).unwrap();
        let field = Array3::ones((10, 10, 10));

        let result = analyzer.detect_plane_wave(&field);
        assert!(result.is_ok());
    }
}
